# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from hunyuan_model.attention_blocks_v2 import FourierEmbedder, Transformer, CrossAttentionDecoder, PointCrossAttentionEncoder
from hunyuan_model.surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from hunyuan_model.volume_decoders import VanillaVolumeDecoder, FlashVDMVolumeDecoding, HierarchicalVolumeDecoding

import trimesh

def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    # try local path
    base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
    model_fld = os.path.expanduser(os.path.join(base_dir, model_path))
    model_path = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
    print(f'Try to load model from local path: {model_path}')
    if not os.path.exists(model_path):
        print('Model path not exists, try to download from huggingface')
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=original_model_path,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=model_fld 
            )
            model_path = os.path.join(path, subfolder)
        except ImportError:
            print(
                "You need to install HuggingFace Hub to load models from the hub."
            )
            raise RuntimeError(f"Model path {model_path} not found")
        except Exception as e:
            raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found")

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant}.{extension}'
    config_path = os.path.join(model_path, 'config.yaml')
    ckpt_path = os.path.join(model_path, ckpt_name)
    return config_path, ckpt_path

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


class VectsetVAE(nn.Module):

    @classmethod
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        num_latents=None,
        pc_size=None,
        pc_sharpedge_size=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        model_kwargs = config['params']
        if num_latents is not None:
            model_kwargs["num_latents"] = num_latents
        if pc_size is not None:
            model_kwargs["pc_size"] = pc_size
        if pc_sharpedge_size is not None:
            model_kwargs["pc_sharpedge_size"] = pc_sharpedge_size
        
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-vae-v2-1',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )
        
    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def __init__(
        self,
        volume_decoder=None,
        surface_extractor=None
    ):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs).abs()
        outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
    ):
        if enabled:
            if adaptive_kv_selection:
                self.volume_decoder = FlashVDMVolumeDecoding(topk_mode)
            else:
                self.volume_decoder = HierarchicalVolumeDecoding()
            if mc_algo not in SurfaceExtractors.keys():
                raise ValueError(f'Unsupported mc_algo {mc_algo}, available:{list(SurfaceExtractors.keys())}')
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class Decoder(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        ckpt_path = None,
    ):

        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        def init_from_ckpt(self, path, ignore_keys=()):
            state_dict = torch.load(path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            keys = list(state_dict.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del state_dict[k]
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")
        

    def decode_latents(self, latents):
        """
        To decode first call the post_kl and the transformer to transform the latents.
        Then call the geo_decoder to predict logits for the queries.
        """
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents


    def decode(self, latents, queries):
        """
        To decode first call the post_kl and the transformer to transform the latents.
        Then call the geo_decoder to predict logits for the queries.
        """
        latents = self.decode_latents(latents)
        logits = self.geo_decoder(queries=queries, latents=latents).abs()   # Changed to abs to remove sign
        return logits



class ShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 2560,
        pc_boundary_size: int = 2560,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        ckpt_path = None,
        has_features: bool = True,
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=self.downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            pc_boundary_size=pc_boundary_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm,
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

        self.has_features = has_features

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)


    def encode(self, surface: torch.Tensor):
        """
        To encode call the encoder with a point cloud.
        Then run the kl divergence and sample from the posterior distribution.
        """
        if self.has_features:
            pc, feats = surface[:, :, :3], surface[:, :, 3:]
            latents, _ = self.encoder(pc, feats)
        else:
            latents, _ = self.encoder(surface, None)

        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

        latents = posterior.sample()
        kl = posterior.kl(dims=(1, 2))
        return kl, latents



    def decode_latents(self, latents):
        """
        To decode first call the post_kl and the transformer to transform the latents.
        Then call the geo_decoder to predict logits for the queries.
        """
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents


    def decode(self, latents, queries):
        """
        To decode first call the post_kl and the transformer to transform the latents.
        Then call the geo_decoder to predict logits for the queries.
        """
        latents = self.decode_latents(latents)
        logits = self.geo_decoder(queries=queries, latents=latents).abs()   # Changed to abs to remove sign
        return logits
    
    def decode_with_grad(self, latents, queries):
        # First pass to compute the UDF
        udf = self.decode(latents, queries)

        # Second pass to compute the gradients
        queries_grad = queries.clone().detach().requires_grad_(True)
        udf_grad = self.decode(latents, queries_grad).flatten()
        grad_outputs = torch.ones_like(udf_grad)
        grads = torch.autograd.grad(
            outputs=udf_grad,
            inputs=queries_grad,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        return udf, grads
    

    def forward(self, pc, queries, with_grads=False, only_encode=False, only_decode=False):
        if only_decode:
            # pc = latents
            o = self.decode(pc, queries).squeeze(-1)
            return {'logits': o}

        kl, latent = self.encode(pc)
        if only_encode:
            return kl, latent
        if with_grads:
            o, g = self.decode_with_grad(latent, queries)
            o = o.squeeze(-1)
            g = g.squeeze(-1)
            return {'logits': o, 'kl': kl, 'grads': g}
        else:
            o = self.decode(latent, queries).squeeze(-1)
            return {'logits': o, 'kl': kl}


