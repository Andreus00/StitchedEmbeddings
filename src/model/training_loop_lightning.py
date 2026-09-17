"""
Lightning training loop of StemNet (StemNet + PanelGroupedAttentionHeads).

Losses: KL (garment + boxmesh encoders), KL-latent (garment <-> boxmesh alignment),
UDF (garment & boxmesh), masked design-parameter loss.
"""

import lightning as pl
import math
import hydra
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from src.util.losses import KLDivergenceLoss
from src.util.design_params_reader import DesignParameters
import trimesh
import os

EXTRACT_VISUALS = True

if EXTRACT_VISUALS:
    try:
        from src.external.MeshUDF.custom_mc.meshudf import get_mesh_from_udf
    except ImportError:
        print("Warning: MeshUDF not found, proceeding without MeshUDF")
        EXTRACT_VISUALS = False


def extract_mesh(model, latent, output_dir=None, garment_name=None, save=True,
                GRID_SIZE=256, coords_range=(-0.8, 0.8), th_alpha=1.05,
                th_beta=1.75, max_batch=2**16, max_dist=0.008, udf_mult=0.01,
                use_fast_grid_filler=True):
    """Extract mesh from latent using marching cubes."""
    def callable_udf_func(x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        udf = model.decode3d(latent.detach(), x).flatten() * udf_mult
        return udf

    vertices, faces = get_mesh_from_udf(
        udf_func=callable_udf_func,
        coords_range=coords_range,
        max_dist=max_dist,
        N=GRID_SIZE,
        use_fast_grid_filler=use_fast_grid_filler,
        differentiable=False,
        th_alpha=th_alpha,
        th_beta=th_beta,
        max_batch=max_batch,
        device='cuda'
    )

    # Account for size of coords range different from [-1, 1]
    range_span = -coords_range[0] + coords_range[1]
    scale = 2 / range_span
    vertices *= scale

    if save and output_dir and garment_name:
        v = vertices.detach().cpu().numpy()
        f = faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=v, faces=f)
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, f"{garment_name}_network_mesh.obj")
        mesh.export(mesh_path, file_type='obj')

    return vertices, faces


@torch.no_grad()
def sided_chamfer(a, b, chunk=4096):
    """Mean squared distance from every point of a (1, N, 3) to its nearest point in b (1, M, 3)."""
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    d2 = torch.cat([torch.cdist(a[i:i + chunk], b).min(dim=1).values ** 2 for i in range(0, a.shape[0], chunk)])
    return d2.mean()


class TrainingLoopLightning(pl.LightningModule):
    def __init__(self, cfg):
        super(TrainingLoopLightning, self).__init__()
        self.cfg = cfg
        self.multi_vae = self.cfg.model
        self.design_parameters: DesignParameters = self.cfg.design_parameters

        self.accum_train_loss = []
        self.accum_val_loss = []
        self.accum_val_ckpt_loss = []

        self.kl_divergence_loss = KLDivergenceLoss()

    def load_state_dict(self, state_dict, strict):
        encoder_state_dict = {
            k.replace(f"multi_vae.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith(f"multi_vae.")
        }
        self.multi_vae.load_state_dict(encoder_state_dict)

    def on_train_epoch_start(self):
        self.multi_vae.set_requires_grad(True)
        self.multi_vae.train()

    def training_step(self, batch, batch_idx):
        """
        Training step with all losses:
        1) Garment -> latent -> UDF + parameters
        2) Boxmesh -> latent -> UDF + parameters
        3) KL loss on latents (normal distribution)
        4) KL latent loss (align garment and boxmesh latents)
        """
        loss = 0

        garment_pcd = batch['garment_pcd'].to(self.device)
        garment_fps = batch['garment_fps'].to(self.device)
        boxmesh_pcd = batch['boxmesh_pcd'].to(self.device)
        boxmesh_fps = batch['boxmesh_fps'].to(self.device)
        gt_parameters = batch['gt_parameters'].to(self.device)
        queries = batch['q'].to(self.device)
        gt_udf = torch.clamp(batch['udf'].to(self.device), min=0, max=self.cfg.max_dist) / self.cfg.max_dist

        # Optional: zero out normals if not passing them
        if not getattr(self.cfg, 'pass_normals', True):
            boxmesh_pcd[:, :, 3:] = 0
            boxmesh_fps[:, :, 3:] = 0

        # ========== ENCODE ==========
        encode_dict = self.multi_vae.encode(garment_pcd, boxmesh_pcd, garment_fps, boxmesh_fps)

        # ========== DECODE ==========
        decode_dict_garm = self.multi_vae.decode(encode_dict['garment_latents'], queries)
        decode_dict_boxmesh = self.multi_vae.decode(encode_dict['boxmesh_latents'], queries)

        # ========== KL LOSSES ==========
        kl_garm_loss = torch.sum(encode_dict['garment_kl']) / encode_dict['garment_kl'].shape[0]
        kl_boxmesh_loss = torch.sum(encode_dict['boxmesh_kl']) / encode_dict['boxmesh_kl'].shape[0]
        kl_loss = kl_garm_loss + kl_boxmesh_loss

        # KL latent alignment loss
        kl_latent_loss = self.kl_divergence_loss(
            encode_dict["boxmesh_latents"],
            encode_dict["garment_latents"]
        )

        # ========== UDF LOSSES ==========
        iou_loss_garment = self.iou_loss(decode_dict_garm['logits'].flatten(), gt_udf.flatten())
        iou_loss_boxmesh = self.iou_loss(decode_dict_boxmesh['logits'].flatten(), gt_udf.flatten())

        udf_garment_loss = torch.nn.functional.mse_loss(decode_dict_garm['logits'].squeeze(-1), gt_udf)
        udf_boxmesh_loss = torch.nn.functional.mse_loss(decode_dict_boxmesh['logits'].squeeze(-1), gt_udf)
        udf_loss = udf_garment_loss + udf_boxmesh_loss

        # ========== PARAMETER LOSSES ==========
        param_loss_dict_garm = getattr(gt_parameters, self.cfg.parameter_loss_function)(
            decode_dict_garm['patterns']['predicted_parameters']
        )
        parameters_loss_garm = getattr(gt_parameters, self.cfg.parameter_aggregate_function)(param_loss_dict_garm)

        param_loss_dict_boxmesh = getattr(gt_parameters, self.cfg.parameter_loss_function)(
            decode_dict_boxmesh['patterns']['predicted_parameters']
        )
        parameters_loss_boxmesh = getattr(gt_parameters, self.cfg.parameter_aggregate_function)(param_loss_dict_boxmesh)

        # Handle dict-style aggregation for both garment and boxmesh separately
        weights = {'select': 0.5, 'select_null': 0.5, 'bool': 1., 'float': 10., 'int': 10.}

        if isinstance(parameters_loss_garm, dict):
            parameters_loss_garm_scalar = 0
            for key, value in parameters_loss_garm.items():
                parameters_loss_garm_scalar += value * weights[key]
                self.log(f"train/param_loss_garm_{key}", value, prog_bar=False, on_step=True, sync_dist=True)
            parameters_loss_garm_scalar = parameters_loss_garm_scalar / len(parameters_loss_garm)
        else:
            parameters_loss_garm_scalar = parameters_loss_garm

        if isinstance(parameters_loss_boxmesh, dict):
            parameters_loss_boxmesh_scalar = 0
            for key, value in parameters_loss_boxmesh.items():
                parameters_loss_boxmesh_scalar += value * weights[key]
                self.log(f"train/param_loss_boxmesh_{key}", value, prog_bar=False, on_step=True, sync_dist=True)
            parameters_loss_boxmesh_scalar = parameters_loss_boxmesh_scalar / len(parameters_loss_boxmesh)
        else:
            parameters_loss_boxmesh_scalar = parameters_loss_boxmesh

        parameters_loss = parameters_loss_garm_scalar + parameters_loss_boxmesh_scalar

        # ========== TOTAL LOSS ==========
        loss += self.cfg.kl_lambda * kl_loss
        loss += self.cfg.kl_latent_lambda * kl_latent_loss
        loss += self.cfg.udf_lambda * udf_loss
        loss += self.cfg.param_lambda * parameters_loss

        # ========== LOGGING ==========
        self.log("train/kl_loss", kl_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/kl_latent_loss", kl_latent_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/udf_loss", udf_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/udf_garment_loss", udf_garment_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/udf_boxmesh_loss", udf_boxmesh_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/parameters_loss", parameters_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/parameters_loss_garm", parameters_loss_garm_scalar, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/parameters_loss_boxmesh", parameters_loss_boxmesh_scalar, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/iou_garment_loss", iou_loss_garment, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/iou_boxmesh_loss", iou_loss_boxmesh, prog_bar=True, on_step=True, sync_dist=True)

        self.accum_train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with all losses."""
        loss = 0

        garment_pcd = batch['garment_pcd'].to(self.device)
        garment_fps = batch['garment_fps'].to(self.device)
        boxmesh_pcd = batch['boxmesh_pcd'].to(self.device)
        boxmesh_fps = batch['boxmesh_fps'].to(self.device)
        gt_parameters = batch['gt_parameters'].to(self.device)
        queries = batch['q'].to(self.device)
        gt_udf = torch.clamp(batch['udf'].to(self.device), min=0, max=self.cfg.max_dist) / self.cfg.max_dist

        # ========== ENCODE ==========
        encode_dict = self.multi_vae.encode(garment_pcd, boxmesh_pcd, garment_fps, boxmesh_fps)

        # ========== DECODE ==========
        decode_dict_garm = self.multi_vae.decode(encode_dict['garment_latents'], queries)
        decode_dict_boxmesh = self.multi_vae.decode(encode_dict['boxmesh_latents'], queries)

        # ========== KL LOSSES ==========
        kl_garm_loss = torch.sum(encode_dict['garment_kl']) / encode_dict['garment_kl'].shape[0]
        kl_boxmesh_loss = torch.sum(encode_dict['boxmesh_kl']) / encode_dict['boxmesh_kl'].shape[0]
        kl_loss = kl_garm_loss + kl_boxmesh_loss

        kl_latent_loss = (
            self.kl_divergence_loss(encode_dict["garment_moments"], encode_dict["boxmesh_moments"]) +
            self.kl_divergence_loss(encode_dict["boxmesh_moments"], encode_dict["garment_moments"])
        ) / 2

        # ========== UDF LOSSES ==========
        iou_loss_garment = self.iou_loss(decode_dict_garm['logits'].flatten(), gt_udf.flatten())
        iou_loss_boxmesh = self.iou_loss(decode_dict_boxmesh['logits'].flatten(), gt_udf.flatten())

        udf_garment_loss = torch.nn.functional.mse_loss(decode_dict_garm['logits'].squeeze(-1), gt_udf)
        udf_boxmesh_loss = torch.nn.functional.mse_loss(decode_dict_boxmesh['logits'].squeeze(-1), gt_udf)
        udf_loss = udf_garment_loss + udf_boxmesh_loss

        # ========== PARAMETER LOSSES ==========
        param_loss_dict_garm = getattr(gt_parameters, self.cfg.parameter_loss_function)(
            decode_dict_garm['patterns']['predicted_parameters']
        )
        parameters_loss_garm = getattr(gt_parameters, self.cfg.parameter_aggregate_function)(param_loss_dict_garm)

        param_loss_dict_boxmesh = getattr(gt_parameters, self.cfg.parameter_loss_function)(
            decode_dict_boxmesh['patterns']['predicted_parameters']
        )
        parameters_loss_boxmesh = getattr(gt_parameters, self.cfg.parameter_aggregate_function)(param_loss_dict_boxmesh)

        # Handle dict-style aggregation
        weights = {
            'select': 0.5,
            'select_null': 0.5,
            'bool': 1.,
            'float': 10.,
            'int': 10.,
        }

        if isinstance(parameters_loss_garm, dict):
            # Convert dict losses to scalars
            parameters_loss_garm_scalar = 0
            for key, value in parameters_loss_garm.items():
                parameters_loss_garm_scalar += value * weights[key]
                self.log(f"val/param_loss_garm_{key}", value, prog_bar=False, on_step=True, sync_dist=True)
            parameters_loss_garm_scalar = parameters_loss_garm_scalar / len(parameters_loss_garm)
        else:
            parameters_loss_garm_scalar = parameters_loss_garm

        if isinstance(parameters_loss_boxmesh, dict):
            # Convert dict losses to scalars
            parameters_loss_boxmesh_scalar = 0
            for key, value in parameters_loss_boxmesh.items():
                parameters_loss_boxmesh_scalar += value * weights[key]
                self.log(f"val/param_loss_boxmesh_{key}", value, prog_bar=False, on_step=True, sync_dist=True)
            parameters_loss_boxmesh_scalar = parameters_loss_boxmesh_scalar / len(parameters_loss_boxmesh)
        else:
            parameters_loss_boxmesh_scalar = parameters_loss_boxmesh

        parameters_loss = parameters_loss_garm_scalar + parameters_loss_boxmesh_scalar

        # ========== TOTAL LOSS ==========
        loss += self.cfg.kl_lambda * kl_loss
        loss += self.cfg.kl_latent_lambda * kl_latent_loss
        loss += self.cfg.udf_lambda * udf_loss
        loss += self.cfg.param_lambda * parameters_loss

        # ========== LOGGING ==========
        self.log("val/kl_loss", kl_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/kl_latent_loss", kl_latent_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/udf_loss", udf_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/udf_garment_loss", udf_garment_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/udf_boxmesh_loss", udf_boxmesh_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/parameters_loss", parameters_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/parameters_loss_garm", parameters_loss_garm_scalar, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/parameters_loss_boxmesh", parameters_loss_boxmesh_scalar, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/iou_garment_loss", iou_loss_garment, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val/iou_boxmesh_loss", iou_loss_boxmesh, prog_bar=True, on_step=True, sync_dist=True)

        self.accum_val_loss.append(loss.item())
        self.accum_val_ckpt_loss.append((udf_loss + parameters_loss).item() if isinstance((udf_loss + parameters_loss), torch.Tensor) else (udf_loss + parameters_loss))

        return loss

    def iou_loss(self, pred, gt, eps=1e-9):
        """Compute IoU loss."""
        iou = torch.sum(torch.min(pred, gt)) / (torch.sum(torch.max(pred, gt)) + eps)
        return iou

    def on_train_epoch_end(self):
        avg_loss = sum(self.accum_train_loss) / len(self.accum_train_loss) if self.accum_train_loss else 0.0
        self.log('avg_train_loss', avg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.accum_train_loss.clear()

    def on_validation_epoch_end(self):
        avg_loss = sum(self.accum_val_loss) / len(self.accum_val_loss) if self.accum_val_loss else 0.0
        self.log('avg_val_loss', avg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.accum_val_loss.clear()

        avg_ckpt_loss = sum(self.accum_val_ckpt_loss) / len(self.accum_val_ckpt_loss) if self.accum_val_ckpt_loss else 0.0
        self.log('avg_val_ckpt_loss', avg_ckpt_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.accum_val_ckpt_loss.clear()

    def set_optimizer(self, optimizer_cfg):
        print(f"Optimizer configuration: {optimizer_cfg}")
        self.optimizer_cfg = optimizer_cfg

    def configure_optimizers(self):
        def get_lambda_lr_scheduler(optimizer, args):
            """Cosine decay with warmup."""
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    base_lr = args.lr * epoch / max(1, args.warmup_epochs)
                else:
                    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                    base_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

                for group in optimizer.param_groups:
                    scale = group.get("lr_scale", 1.0)
                    group["lr"] = base_lr * scale

                return base_lr

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        def get_exponential_lr_scheduler(optimizer, args):
            return ExponentialLR(optimizer=optimizer, gamma=args.gamma)

        empty_tensor = [torch.empty(0)]
        optimizer = hydra.utils.instantiate(self.optimizer_cfg.optimizer, params=empty_tensor)
        optimizer = self.multi_vae.add_param_groups_to_optimizer(optimizer, self.optimizer_cfg.param_group_lr)

        if self.optimizer_cfg.scheduler_cfg.type == "sinusoidal":
            scheduler = get_lambda_lr_scheduler(optimizer, self.optimizer_cfg.scheduler_cfg)
        elif self.optimizer_cfg.scheduler_cfg.type == "exponential":
            scheduler = get_exponential_lr_scheduler(optimizer, self.optimizer_cfg.scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.optimizer_cfg.scheduler_cfg.frequency
            }
        }
