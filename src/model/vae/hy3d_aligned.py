import torch

import torch.nn as nn

import src.external.Hunyuan3D.models_ae as models_ae
from typing import Optional, Union, List
import os
import numpy as np
from src.model.vae.hy3d_pattern_classifier import EncoderWithFps
import torch.nn.functional as F
from einops import rearrange

scaled_dot_product_attention = F.scaled_dot_product_attention
if os.environ.get('CA_USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


def fps(
    src: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    ratio: Optional[Union[torch.Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[torch.Tensor, List[int]]] = None,
):
    src = src.float()
    from torch_cluster import fps as fps_fn
    output = fps_fn(src, batch, ratio, random_start)
    return output

class CrossAttentionProcessor:
    def __call__(self, attn, q, k, v):
        out = scaled_dot_product_attention(q, k, v)
        return out


class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                out = scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::50, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = torch.mean(sim, -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
        topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=topk_ind)
        k0 = torch.gather(k, dim=-2, index=topk_ind)
        return k0, v0


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::30, :]
        sim = q1 @ k.transpose(-1, -2)
        # sim = sim.to(torch.float32)
        sim = sim.softmax(-1)
        sim = torch.mean(sim, 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = torch.unique(activated_token, return_counts=True)[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        index = index.expand(-1, v.shape[1], -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=index)
        k0 = torch.gather(k, dim=-2, index=index)
        return k0, v0

class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dim = input_dim

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
        self, *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                print('Save kv cache,this should be called only once for one mesh')
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        
        out = scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        """
        Initialize a diagonal Gaussian distribution with mean and log-variance parameters.

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]): 
                Either a single tensor containing concatenated mean and log-variance along `feat_dim`,
                or a list of two tensors [mean, logvar].
            deterministic (bool, optional): If True, the distribution is deterministic (zero variance). 
                Default is False. feat_dim (int, optional): Dimension along which mean and logvar are 
                concatenated if parameters is a single tensor. Default is 1.
        """
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        """
        Sample from the diagonal Gaussian distribution.

        Returns:
            torch.Tensor: A sample tensor with the same shape as the mean.
        """
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2, 3)):
        """
        Compute the Kullback-Leibler (KL) divergence between this distribution and another.

        If `other` is None, compute KL divergence to a standard normal distribution N(0, I).

        Args:
            other (DiagonalGaussianDistribution, optional): Another diagonal Gaussian distribution.
            dims (tuple, optional): Dimensions along which to compute the mean KL divergence. 
                Default is (1, 2, 3).

        Returns:
            torch.Tensor: The mean KL divergence value.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

import copy


class HunyuanMeshAligned(nn.Module):
    def __init__(self, 
                latent_query_num,
                use_hy3d=False,
                importance_percentage: float = 0.,
                boundary_percentage: float = 0.25,
                ckpt: str = None,
                init_ckpt: str = None,
                ):
        super(HunyuanMeshAligned, self).__init__()

        num_latents = latent_query_num
        self.importance_percentage = importance_percentage
        self.boundary_percentage = boundary_percentage
        self.hunyuan_model = models_ae.__dict__['hunyuan_garments'](sharpedge_ratio=self.importance_percentage, boundary_ratio=self.boundary_percentage)
        # Training initialisation of encoders / decoder: the original Hunyuan3D-2.1 shape-VAE weights
        # (hunyuan3d-vae-v2-1/model.fp16.ckpt, a plain state dict) or a fine-tune ({'model': state_dict}).
        # Not needed for inference: the StemNet checkpoint holds all of these weights.
        if init_ckpt is not None and os.path.exists(init_ckpt):
            print(f"Initialising the Hunyuan3D shape VAE from {init_ckpt}")
            state_dict = torch.load(init_ckpt, map_location='cpu', weights_only=False)
            state_dict = state_dict.get('model', state_dict.get('state_dict', state_dict))
            self.hunyuan_model.load_state_dict({k: v.float() for k, v in state_dict.items()}, strict=True)
        elif init_ckpt is not None:
            print(f"WARNING: Hunyuan3D init checkpoint {init_ckpt} not found - encoders start from random weights")
        downsample_ratio = self.hunyuan_model.encoder.downsample_ratio
        num_points = num_latents * downsample_ratio
        self.num_imp_pts = int(num_points * self.importance_percentage)
        self.num_bnd_pts = int(num_points * self.boundary_percentage)
        self.num_rnd_pts = num_points - (self.num_bnd_pts + self.num_imp_pts)

        self.hunyuan_model = self.hunyuan_model.to(torch.float32)
        # self.hunyuan_model.eval()
        self.hunyuan_model.requires_grad_(False)
        for param in self.hunyuan_model.parameters():
            param.requires_grad = False

        self.hunyuan_model.encoder.pc_sharpedge_size = self.num_imp_pts
        if self.num_bnd_pts is not None:
            self.hunyuan_model.encoder.pc_boundary_size = self.num_bnd_pts
        self.hunyuan_model.encoder.pc_size = self.num_rnd_pts
        self.hunyuan_model.encoder.num_latents = num_latents
        self.hunyuan_model.encoder.downsample_ratio = downsample_ratio

        self.source_encoder = EncoderWithFps(self.hunyuan_model) # EncoderWithoutFps(self.hunyuan_model)  # encoder to align
        for param in self.source_encoder.parameters():
            param.requires_grad = True
        self.target_encoder = EncoderWithFps(self.hunyuan_model) # EncoderWithoutFps(self.hunyuan_model)  # encoder to align

        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location="cuda", weights_only=False)["state_dict"]
            self.load_state_dict(state_dict, special_load=True)


    # Old loader for the boxmesh -> simulation encoder.
    # def load_state_dict(self, state_dict, prefix="model.boxmesh_encoder.", special_load=False):
    def load_state_dict(self, state_dict, prefix="model.boxmesh_encoder.", special_load=False):
        # if ckpt_path.endswith(".ckpt"): # lightning
        # checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        # state_dict = checkpoint.get("state_dict", checkpoint)
        if special_load:
            state_dict = {
                k.replace(f"{prefix}", "", 1): v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix}")
            }
            self.source_encoder.load_state_dict(state_dict)
        else:
            super().load_state_dict(state_dict)
        
        
    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.source_encoder.train()
        # self.target_encoder.eval()

    def run_kl(self, moments):
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

        latents = posterior.sample()
        kl = posterior.kl(dims=(1, 2))
        return kl, latents

    def decode(self, latents, q):
        latents = self.hunyuan_model.decode(latents, q).reshape(latents.shape[0], -1).abs()
        return latents

    def encode_target_and_source(self,
            target_surface,
            source_surface,
            target_fps,
            source_fps,
            ):
        # encode
        source_latents = self.source_encoder(source_surface, source_fps)
        with torch.no_grad():
            target_latents = self.target_encoder(target_surface, target_fps)
        return {
            "source_latents": source_latents,
            "target_latents": target_latents,
        }


    def forward(self,
            target_surface,
            source_surface,
            target_fps,
            source_fps,
            ):
        return self.encode_target_and_source(
                    target_surface,
                    source_surface,
                    target_fps,
                    source_fps,
                    )