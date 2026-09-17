"""
StemNet: garment/boxmesh encoders -> aligned latent -> UDF decoder + sewing-pattern heads.

Features:
- Supports both latent_code and discriminative_latent as prediction space
- Panel-grouped attention heads for parameter prediction
- Compatible with GarmentCodeData dataset (with boxmesh)
- All losses: KL, KL-latent, UDF, parameter, RDM
"""

from src.model.vae.hy3d_aligned import DiagonalGaussianDistribution
import copy
import torch
import torch.nn as nn
from src.util.design_params_reader import DesignParameters
import omegaconf


class StemNet(nn.Module):
    def __init__(self,
                box_to_sim_model,
                sim_to_pattern_model,
                design_parameters,
                use_discriminative_latents=True,  # NEW: toggle for prediction space
                tmp_folder="./tmp/"
                ):
        """
        StemNet.

        Args:
            box_to_sim_model: Model for boxmesh -> latent
            sim_to_pattern_model: Model for latent -> patterns
            design_parameters: DesignParameters instance
            use_discriminative_latents: If True, predict from discriminative_latents.
                                       If False, predict from latent_code directly.
            tmp_folder: Temporary folder for intermediate files
        """
        super(StemNet, self).__init__()

        self.design_parameters: DesignParameters = design_parameters
        self.tmp_folder = tmp_folder
        self.use_discriminative_latents = use_discriminative_latents

        # Decoder for Latent -> pattern (with PanelGroupedAttentionHeads)
        self.sim_to_pattern_model = sim_to_pattern_model

        # Encoder for Garment Mesh -> Latent
        self.garment_encoder = box_to_sim_model.target_encoder

        # Encoder for BoxMesh -> Garment Latent
        self.box_encoder = box_to_sim_model.source_encoder

        # Latent to 3D decoder (produces discriminative_latents)
        self.post_kl = box_to_sim_model.hunyuan_model.post_kl
        self.transformer = box_to_sim_model.hunyuan_model.transformer
        self.geo_decoder = box_to_sim_model.hunyuan_model.geo_decoder

        self.set_requires_grad(True)

    def set_requires_grad(self, mode):
        for param in self.parameters():
            param.requires_grad = mode

    def run_kl(self, moments, deterministic=False):
        """Sample latents from moments and compute KL divergence."""
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1, deterministic=deterministic)
        latents = posterior.sample()
        kl = posterior.kl(dims=(1, 2))
        return kl, latents

    def encode(self, garment_surface, boxmesh_surface, garment_fps=None, boxmesh_fps=None):
        """
        Encode both garment and boxmesh point clouds.

        Returns:
            dict with keys: garment_moments, garment_kl, garment_latents,
                           boxmesh_moments, boxmesh_kl, boxmesh_latents
        """
        # Encode garment
        garment_moments = self.garment_encoder(garment_surface, garment_fps)
        kl_garment, garment_latents = self.run_kl(garment_moments)

        # Encode boxmesh
        boxmesh_moments = self.box_encoder(boxmesh_surface, boxmesh_fps)
        kl_boxmesh, boxmesh_latents = self.run_kl(boxmesh_moments)

        return {
            "garment_moments": garment_moments,
            "garment_kl": kl_garment,
            "garment_latents": garment_latents,
            "boxmesh_moments": boxmesh_moments,
            "boxmesh_kl": kl_boxmesh,
            "boxmesh_latents": boxmesh_latents,
        }

    def decode_latents(self, latents):
        """
        Transform latents to discriminative latents.

        Args:
            latents: (B, N, D) latent codes

        Returns:
            discriminative_latents: (B, N, D) transformed latents
        """
        latents = self.post_kl(latents)
        discriminative_latents = self.transformer(latents)
        return discriminative_latents

    def predict_udf(self, latents, queries):
        """
        Predict UDF values from latents (discriminative or direct).

        Args:
            latents: (B, N, D) either latent_code or discriminative_latents
            queries: (B, Q, 3) query points

        Returns:
            logits: (B, Q, 1) UDF predictions
        """
        logits = self.geo_decoder(queries=queries, latents=latents).abs()
        return logits

    # Alias for compatibility with close_loop_v2.extract_mesh
    def predict_logits(self, latents, queries):
        return self.predict_udf(latents, queries)

    def classify(self, latents):
        """
        Predict pattern parameters from latents.
        Uses PanelGroupedAttentionHeads internally.

        Args:
            latents: (B, N, D) either latent_code or discriminative_latents

        Returns:
            dict with 'predicted_parameters' and optionally 'panel_features'
        """
        return self.sim_to_pattern_model(latents)

    def classify_latents(self, latents):
        """
        Predict pattern parameters from raw latents (mirrors MultimodalModelClassifyV2 API).
        Handles discriminative latent decoding internally.

        Args:
            latents: (B, N, D) raw latent codes from the encoder

        Returns:
            dict with 'predicted_parameters' and optionally 'panel_features'
        """
        discriminative_latents = self.decode_latents(latents)
        prediction_latents = discriminative_latents if self.use_discriminative_latents else latents
        return self.sim_to_pattern_model(prediction_latents)

    def decode(self, latents, queries):
        """
        Full decode: latents -> UDF + patterns.

        Args:
            latents: (B, N, D) latent codes
            queries: (B, Q, 3) query points for UDF

        Returns:
            dict with keys: logits, patterns, discriminative_latents
        """
        # Get discriminative latents
        discriminative_latents = self.decode_latents(latents)

        # Choose prediction space based on configuration
        if self.use_discriminative_latents:
            prediction_latents = discriminative_latents
        else:
            prediction_latents = latents

        # Predict UDF (always from discriminative latents for 3D)
        logits = self.predict_udf(discriminative_latents, queries)

        # Predict patterns (from configured space)
        patterns = self.classify(prediction_latents)

        return {
            'logits': logits,
            'patterns': patterns,
            'discriminative_latents': discriminative_latents,
        }

    def decode3d(self, latents, queries):
        """
        Decode only 3D UDF (for mesh extraction).

        Args:
            latents: (B, N, D) latent codes
            queries: (B, Q, 3) query points

        Returns:
            logits: (B, Q, 1) UDF predictions
        """
        discriminative_latents = self.decode_latents(latents)
        logits = self.predict_udf(discriminative_latents, queries)
        return logits

    def add_param_groups_to_optimizer(self, optimizer: torch.optim.Optimizer,
                                     param_group_lr: omegaconf.dictconfig.DictConfig):
        """Add all model parameters to optimizer."""
        optimizer.add_param_group({
            'params': [p for p in self.parameters() if p.requires_grad]
        })
        return optimizer
