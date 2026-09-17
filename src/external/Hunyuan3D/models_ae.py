from functools import wraps

import os
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from timm.models.layers import DropPath

from src.external.Hunyuan3D.hunyuan_model.model import ShapeVAE
from src.external.Hunyuan3D.hunyuan_model.model_v2 import ShapeVAE as ShapeVAE_v2

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        ###### fps
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return None, x


    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        _, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o, 'kl': None}
    

class AutoEncoderV2(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        ###### fps
        flattened_rnd = pc[:, :N//2, :]
        flattened_imp = pc[:, N//2:, :]
        flattened_rnd = flattened_rnd.reshape(B*(N//2), D)
        flattened_imp = flattened_imp.reshape(B*(N//2), D)

        batch = torch.arange(B).to(pc.device)
        batch_rnd = torch.repeat_interleave(batch, (N//2))
        batch_imp = torch.repeat_interleave(batch, (N//2))

        pos_rnd = flattened_rnd
        pos_imp = flattened_imp

        ratio = 1.0 * self.num_latents / self.num_inputs
        idx_rnd = fps(pos_rnd, batch_rnd, ratio=ratio)
        idx_imp = fps(pos_imp, batch_imp, ratio=ratio)

        sampled_pc_rnd = pos_rnd[idx_rnd]
        sampled_pc_imp = pos_imp[idx_imp]
        
        sampled_pc = torch.cat([sampled_pc_rnd, sampled_pc_imp], dim=1).view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return None, x


    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        _, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o, 'kl': None}

class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)


    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        ###### fps
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x


    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):

        kl, x = self.encode(pc)
        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o, 'kl': kl}
    

class KLAutoEncoderV2(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        compute_grad=True
    ):
        super().__init__()

        self.compute_grad = compute_grad

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)


    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        n_surf_bnd_pts = N // 4
        n_surf_imp_pts = N // 4
        n_surf_rnd_pts = N - (n_surf_bnd_pts + n_surf_imp_pts)
        
        ###### separate fps for random and importance points
        flattened_rnd = pc[:, :n_surf_rnd_pts, :]
        flattened_imp = pc[:, n_surf_rnd_pts:n_surf_rnd_pts+n_surf_imp_pts, :]
        flattened_bnd = pc[:, n_surf_rnd_pts+n_surf_imp_pts:, :]
        flattened_rnd = flattened_rnd.reshape(B*(n_surf_rnd_pts), D)
        flattened_imp = flattened_imp.reshape(B*(n_surf_imp_pts), D)
        flattened_bnd = flattened_bnd.reshape(B*(n_surf_bnd_pts), D)

        batch = torch.arange(B).to(pc.device)
        batch_rnd = torch.repeat_interleave(batch, n_surf_rnd_pts)
        batch_imp = torch.repeat_interleave(batch, n_surf_imp_pts)
        batch_bnd = torch.repeat_interleave(batch, n_surf_bnd_pts)

        pos_rnd = flattened_rnd
        pos_imp = flattened_imp
        pos_bnd = flattened_bnd

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx_rnd = fps(pos_rnd, batch_rnd, ratio=ratio)
        idx_imp = fps(pos_imp, batch_imp, ratio=ratio)
        idx_bnd = fps(pos_bnd, batch_bnd, ratio=ratio)


        sampled_pc_rnd = pos_rnd[idx_rnd].reshape(B, -1, 3)
        sampled_pc_imp = pos_imp[idx_imp].reshape(B, -1, 3)
        sampled_pc_bnd = pos_bnd[idx_bnd].reshape(B, -1, 3)
#        print(sampled_pc_rnd.shape, sampled_pc_bnd.shape,sampled_pc_imp.shape)
        sampled_pc = torch.cat([sampled_pc_rnd, sampled_pc_imp, sampled_pc_bnd], dim=1).view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x


    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)
    
    def decode_with_grad(self, x, queries):
        # First pass to compute the UDF
        udf = self.decode(x, queries)

        # Second pass to compute the gradients
        queries_grad = queries.clone().detach().requires_grad_(True)
        udf_grad = self.decode(x, queries_grad).flatten()
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

    def forward(self, pc, queries, with_grads=True, only_encode=False, only_decode=False):
        if only_decode:
            o = self.decode(pc, queries).squeeze(-1)
            return {'logits': o}



        kl, x = self.encode(pc)
        if only_encode:
            return kl, x
        if with_grads:
            o, g = self.decode_with_grad(x, queries)
            o = o.squeeze(-1)
            g = g.squeeze(-1)
            return {'logits': o, 'kl': kl, 'grads': g}
        else:
            o = self.decode(x, queries).squeeze(-1)
            return {'logits': o, 'kl': kl}

def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False, v2=False):
    if determinisitc:
        if v2:
            model = AutoEncoderV2(
                depth=24,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                heads = 8,
                dim_head = 64,
            )
        else:
            model = AutoEncoder(
                depth=24,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                heads = 8,
                dim_head = 64,
            )
    else:
        if v2:
            model = KLAutoEncoderV2(
                depth=24,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                latent_dim = latent_dim,
                heads = 8,
                dim_head = 64,
            )
        else:
            model = KLAutoEncoder(
                depth=24,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                latent_dim = latent_dim,
                heads = 8,
                dim_head = 64,
            )
    return model


def ae_garments(N=8192):
    return create_autoencoder(dim=512, M=1024, latent_dim=8, N=N, determinisitc=True, v2=True)

# def kl_garments(N=8192):
#     return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)


# test cluster
def kl_garments(N=8192, M=512, D=32):
    return create_autoencoder(dim=512, M=M, latent_dim=D, N=N, determinisitc=False, v2=True)



HUNYUAN_VAE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'hunyuan3d-vae-v2-1.yaml')


def hunyuan_garments(N=4096*10*2, M=4096, D=64, sharpedge_ratio=0.5, boundary_ratio=None):
    """Hunyuan3D-2.1 shape VAE architecture (hunyuan3d-vae-v2-1 hyper-parameters), randomly initialised.
    Weights are NOT loaded here: the StemNet checkpoint holds all of them at inference, and for training
    the original Hunyuan3D weights (or a fine-tune) are loaded through HunyuanMeshAligned(init_ckpt=...)."""
    import yaml
    model_class = ShapeVAE if boundary_ratio is None else ShapeVAE_v2
    with open(HUNYUAN_VAE_CONFIG) as f:
        kwds = yaml.safe_load(f)['params']
    kwds["num_latents"] = M
    if boundary_ratio is not None:
        kwds["pc_size"] = int(N * (1 - (sharpedge_ratio + boundary_ratio)))
        kwds["pc_sharpedge_size"] = int(N * sharpedge_ratio)
        kwds["pc_boundary_size"] = int(N * boundary_ratio)
    else:
        kwds["pc_size"] = int(N * (1 - sharpedge_ratio))
        kwds["pc_sharpedge_size"] = int(N * sharpedge_ratio)

    vae = model_class(**kwds).to(torch.float32)
    assert vae.latent_shape[0] == M, f"{vae.latent_shape[0]} != {M}"
    assert vae.latent_shape[1] == D, f"{vae.latent_shape[1]} != {D}"
    assert vae.downsample_ratio * vae.latent_shape[0] == N, f"{vae.downsample_ratio * vae.latent_shape} != {N}"
    return vae

# test local
# def kl_garments(N=8192):
#     return create_autoencoder(dim=512, M=2048, latent_dim=16, N=N, determinisitc=False, v2=True)

###############


def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, determinisitc=False)
    
def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)

def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)

def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, determinisitc=False)

def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)

def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, determinisitc=False)

def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, determinisitc=False)

def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, determinisitc=False)

###
def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, determinisitc=True)

def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, determinisitc=True)

def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, determinisitc=True)

def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, determinisitc=True)

###
def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, determinisitc=True)

def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, determinisitc=True)

def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, determinisitc=True)
