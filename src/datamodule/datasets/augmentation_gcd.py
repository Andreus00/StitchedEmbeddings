"""
Augmentation strategies for the GarmentCodeData dataset (GCD_ALL).

All augmentations operate on a sample dict with keys:
    garment_pcd:          (N, 7) float32 — XYZ + normals + feature
    garment_fps:          (F, 7) float32 — farthest-point sub-sample
    boxmesh_pcd:          (N, 7) float32 — XYZ + normals + feature (boxmesh)
    boxmesh_fps:          (F, 7) float32 — farthest-point sub-sample (boxmesh)
    q:                    (M, 3) float32 — UDF query points
    udf:                  (M,)   float32 — unsigned distance values
    gt_parameters:        DesignParameters — sewing pattern ground truth
    gt_parameters_masked: (D,)   float32 — flat GT parameter vector (with active mask)

Spatial CutMix requires a second sample from the dataset.
Mirrors the interface of augmentation_nt.NTAugmentation.
"""

import copy

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Panel taxonomy  (matches GarmentCodeData panel names)
# ──────────────────────────────────────────────────────────────────────────────

UPPER_PANELS = frozenset([
    "top_front", "top_back", "top_front_left",
    "sleeve_lf", "sleeve_lb", "sleeve_rf", "sleeve_rb",
    "hood_left", "hood_right",
])

LOWER_PANELS = frozenset([
    "skirt_front", "skirt_back",
    "skirt_front_left", "skirt_front_right",
    "skirt_back_left", "skirt_back_right",
    "skirt_left", "skirt_right",
    "pant_front_left", "pant_front_right",
    "pant_back_left", "pant_back_right",
    "wb_front", "wb_back",
])


# ──────────────────────────────────────────────────────────────────────────────
# Point-cloud augmentations  (same primitives as augmentation_nt)
# ──────────────────────────────────────────────────────────────────────────────

def _pcd_add_noise(pcd: torch.Tensor, xyz_std: float = 0.005) -> torch.Tensor:
    """Add Gaussian noise to XYZ positions (columns 0-2).  pcd: (N, C)."""
    pcd = pcd.clone()
    pcd[:, :3] += torch.randn(pcd.shape[0], 3, dtype=pcd.dtype) * xyz_std
    return pcd


def _pcd_jitter_normals(pcd: torch.Tensor, normal_std: float = 0.01) -> torch.Tensor:
    """Jitter normal vectors (columns 3-5) and re-normalise.  pcd: (N, C)."""
    pcd = pcd.clone()
    pcd[:, 3:6] += torch.randn(pcd.shape[0], 3, dtype=pcd.dtype) * normal_std
    norms = pcd[:, 3:6].norm(dim=1, keepdim=True).clamp(min=1e-6)
    pcd[:, 3:6] = pcd[:, 3:6] / norms
    return pcd


def _pcd_random_dropout(
    pcd: torch.Tensor,
    target_n: int,
    dropout_rate: float = 0.05,
) -> torch.Tensor:
    """
    Randomly discard `dropout_rate` fraction of points, then resample back
    to `target_n` rows so the output size stays fixed for batching.
    """
    N = pcd.shape[0]
    keep_n = max(1, int(N * (1.0 - dropout_rate)))
    keep_idx = torch.randperm(N, device=pcd.device)[:keep_n]
    pcd_dropped = pcd[keep_idx]
    if pcd_dropped.shape[0] >= target_n:
        idx = torch.randperm(pcd_dropped.shape[0], device=pcd.device)[:target_n]
    else:
        idx = torch.randint(0, pcd_dropped.shape[0], (target_n,), device=pcd.device)
    return pcd_dropped[idx]


def _pcd_random_scale(
    pcd: torch.Tensor,
    fps: torch.Tensor,
    q: torch.Tensor,
    udf: torch.Tensor,
    scale_range: tuple = (0.95, 1.05),
):
    """Isotropic scaling of positions, queries, and UDF distances."""
    lo, hi = scale_range
    scale = lo + (hi - lo) * torch.rand(1).item()
    pcd = pcd.clone(); pcd[:, :3] *= scale
    fps = fps.clone(); fps[:, :3] *= scale
    q = q * scale
    udf = udf * scale
    return pcd, fps, q, udf


def _pcd_random_rotation_y(
    pcd: torch.Tensor,
    fps: torch.Tensor,
    q: torch.Tensor,
):
    """Rotate positions + normals around the vertical (Y) axis."""
    angle = float(torch.empty(1).uniform_(0.0, 2.0 * np.pi))
    c, s = np.cos(angle), np.sin(angle)
    R = torch.tensor(
        [[c,   0.0,  s],
         [0.0, 1.0,  0.0],
         [-s,  0.0,  c]],
        dtype=pcd.dtype,
    )
    pcd = pcd.clone()
    pcd[:, :3] = pcd[:, :3] @ R.T
    pcd[:, 3:6] = pcd[:, 3:6] @ R.T
    fps = fps.clone()
    fps[:, :3] = fps[:, :3] @ R.T
    fps[:, 3:6] = fps[:, 3:6] @ R.T
    q = q @ R.T
    return pcd, fps, q


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for Spatial CutMix
# ──────────────────────────────────────────────────────────────────────────────

def _resample_to(tensor: torch.Tensor, target_n: int) -> torch.Tensor:
    """Resample tensor along dim-0 to exactly `target_n` rows."""
    n = tensor.shape[0]
    if n == 0:
        raise ValueError("Cannot resample an empty tensor.")
    if n >= target_n:
        idx = torch.randperm(n, device=tensor.device)[:target_n]
    else:
        idx = torch.randint(0, n, (target_n,), device=tensor.device)
    return tensor[idx]


def _mix_pcd_by_ycut(pcd_a, fps_a, pcd_b, fps_b, y_cut, min_half_points):
    """
    Mix upper half (Y >= y_cut) from a with lower half (Y < y_cut) from b.
    Returns (pcd_mixed, fps_mixed) or (None, None) if too few points.
    """
    mask_up = pcd_a[:, 1] >= y_cut
    mask_lo = pcd_b[:, 1] < y_cut
    if mask_up.sum() < min_half_points or mask_lo.sum() < min_half_points:
        return None, None

    pcd_mixed = _resample_to(
        torch.cat([pcd_a[mask_up], pcd_b[mask_lo]], dim=0),
        pcd_a.shape[0],
    )

    mask_fps_up = fps_a[:, 1] >= y_cut
    mask_fps_lo = fps_b[:, 1] < y_cut
    fps_cat = torch.cat(
        [p for p in [fps_a[mask_fps_up], fps_b[mask_fps_lo]] if len(p) > 0], dim=0
    )
    fps_mixed = _resample_to(fps_cat, fps_a.shape[0]) if len(fps_cat) > 0 else fps_a

    return pcd_mixed, fps_mixed


def _mix_design_parameters(gt_a, gt_b):
    """
    Return a new DesignParameters where:
      - upper-body physical panels  → values from gt_a
      - lower-body physical panels  → values from gt_b
      - meta existence flags        → upper flags from gt_a, lower flags from gt_b
    """
    mixed = copy.deepcopy(gt_a)

    for panel_name in LOWER_PANELS:
        if panel_name not in mixed.panel_data or panel_name not in gt_b.panel_data:
            continue
        for param_name, param in mixed.panel_data[panel_name].items():
            b_val = gt_b.panel_data[panel_name][param_name].get_value()
            param.set_value(b_val.clone() if isinstance(b_val, torch.Tensor) else b_val)

    if "meta" in mixed.panel_data and "meta" in gt_b.panel_data:
        for param_name, param in mixed.panel_data["meta"].items():
            if param_name in LOWER_PANELS:
                b_val = gt_b.panel_data["meta"][param_name].get_value()
                param.set_value(b_val.clone() if isinstance(b_val, torch.Tensor) else b_val)

    return mixed


def _dp_to_flat_numpy_masked(dp, template) -> np.ndarray:
    """
    Recompute gt_parameters_masked from an already-loaded DesignParameters object.
    Mirrors extract_gt_design_params_with_mask in gcd_all.py but accepts dp directly
    (rather than a file path) so it can be used after CutMix mixing.
    """
    predictions = {}
    for panel_name, panel_params in dp.panel_data.items():
        predictions[panel_name] = {}
        for param_name, param in panel_params.items():
            predictions[panel_name][param_name] = param.get_value()

    active_mask = dp.get_active_mask(predictions)

    flat = []
    for panel_name in sorted(template.panel_data.keys()):
        panel_params = template.panel_data[panel_name]
        for param_name in sorted(panel_params.keys()):
            template_param = panel_params[param_name]
            out_size = template_param.get_output_size()
            is_active = active_mask.get(panel_name, {}).get(param_name, False)
            if is_active and panel_name in dp.panel_data and param_name in dp.panel_data.get(panel_name, {}):
                val = template_param.value_of(dp.panel_data[panel_name][param_name])
                if isinstance(val, torch.Tensor):
                    arr = val.detach().cpu().numpy().flatten()
                else:
                    arr = np.array([val]).flatten()
                if arr.size > out_size:
                    arr = arr[:out_size]
                elif arr.size < out_size:
                    arr = np.pad(arr, (0, out_size - arr.size))
            else:
                arr = np.zeros(out_size, dtype=np.float32)
            flat.append(arr)

    return np.concatenate(flat, axis=0).astype(np.float32) if flat else np.array([], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial CutMix
# ──────────────────────────────────────────────────────────────────────────────

def spatial_cutmix(
    sample_a: dict,
    sample_b: dict,
    design_params_template,
    min_half_points: int = 64,
) -> dict:
    """
    Mix the upper-body region (Y >= y_cut) from sample_a with the lower-body
    region (Y < y_cut) from sample_b. The cut plane is drawn uniformly from
    the waist region [-0.15, 0.15] in body-normalised coordinates.

    Applies to: garment_pcd, garment_fps, boxmesh_pcd, boxmesh_fps,
                q, udf, gt_parameters, gt_parameters_masked.
    Unaffected: garment_name (kept from sample_a).

    Returns sample_a unchanged if either garment half has fewer than min_half_points.
    """
    y_cut = float(np.random.uniform(-0.15, 0.15))

    # ── Garment point cloud ───────────────────────────────────────────────────
    garment_pcd_mixed, garment_fps_mixed = _mix_pcd_by_ycut(
        sample_a["garment_pcd"], sample_a["garment_fps"],
        sample_b["garment_pcd"], sample_b["garment_fps"],
        y_cut, min_half_points,
    )
    if garment_pcd_mixed is None:
        return sample_a

    # ── Boxmesh point cloud (apply same y_cut) ────────────────────────────────
    has_boxmesh = "boxmesh_pcd" in sample_a and "boxmesh_pcd" in sample_b
    if has_boxmesh:
        boxmesh_pcd_mixed, boxmesh_fps_mixed = _mix_pcd_by_ycut(
            sample_a["boxmesh_pcd"], sample_a["boxmesh_fps"],
            sample_b["boxmesh_pcd"], sample_b["boxmesh_fps"],
            y_cut, min_half_points=1,  # relaxed — boxmesh is sparser
        )
        if boxmesh_pcd_mixed is None:
            boxmesh_pcd_mixed = sample_a["boxmesh_pcd"]
            boxmesh_fps_mixed = sample_a["boxmesh_fps"]

    # ── Queries + UDF ─────────────────────────────────────────────────────────
    q_a, udf_a = sample_a["q"], sample_a["udf"]
    q_b, udf_b = sample_b["q"], sample_b["udf"]
    mask_q_up = q_a[:, 1] >= y_cut
    mask_q_lo = q_b[:, 1] < y_cut
    q_cat   = torch.cat([q_a[mask_q_up],   q_b[mask_q_lo]],   dim=0)
    udf_cat = torch.cat([udf_a[mask_q_up], udf_b[mask_q_lo]], dim=0)
    if len(q_cat) > 0:
        idx_q     = _resample_to(torch.arange(len(q_cat)), q_a.shape[0])
        q_mixed   = q_cat[idx_q]
        udf_mixed = udf_cat[idx_q]
    else:
        q_mixed, udf_mixed = q_a, udf_a

    # ── Design parameters ─────────────────────────────────────────────────────
    gt_mixed        = _mix_design_parameters(sample_a["gt_parameters"], sample_b["gt_parameters"])
    gt_masked_mixed = torch.from_numpy(_dp_to_flat_numpy_masked(gt_mixed, design_params_template))

    mixed = dict(sample_a)
    mixed["garment_pcd"]          = garment_pcd_mixed
    mixed["garment_fps"]          = garment_fps_mixed
    mixed["q"]                    = q_mixed
    mixed["udf"]                  = udf_mixed
    mixed["gt_parameters"]        = gt_mixed
    mixed["gt_parameters_masked"] = gt_masked_mixed
    if has_boxmesh:
        mixed["boxmesh_pcd"] = boxmesh_pcd_mixed
        mixed["boxmesh_fps"] = boxmesh_fps_mixed
    return mixed


# ──────────────────────────────────────────────────────────────────────────────
# GCDAugmentation — modular container
# ──────────────────────────────────────────────────────────────────────────────

class GCDAugmentation:
    """
    Modular augmentation pipeline for the GCD_ALL dataset.

    Mirrors NTAugmentation but also augments boxmesh_pcd / boxmesh_fps in
    parallel with garment_pcd / garment_fps (same transform applied to both).

    Expected configuration block (OmegaConf / plain dict):

        augmentation:
          cutmix:
            enabled: true
            prob: 0.5
          xyz_noise:
            enabled: true
            std: 0.005
          normal_jitter:
            enabled: true
            std: 0.01
          dropout:
            enabled: false
            rate: 0.05
          scale:
            enabled: true
            range: [0.97, 1.03]
          rotation_y:
            enabled: false
    """

    def __init__(self, cfg, design_parameters):
        def _get(key, sub, default):
            block = cfg.get(key, {}) if hasattr(cfg, "get") else getattr(cfg, key, {})
            if hasattr(block, "get"):
                return block.get(sub, default)
            return getattr(block, sub, default)

        self.design_parameters   = design_parameters

        self.use_cutmix          = _get("cutmix",        "enabled", False)
        self.cutmix_prob         = _get("cutmix",        "prob",    0.5)
        self.use_xyz_noise       = _get("xyz_noise",     "enabled", False)
        self.xyz_noise_std       = _get("xyz_noise",     "std",     0.005)
        self.use_norm_jitter     = _get("normal_jitter", "enabled", False)
        self.norm_jitter_std     = _get("normal_jitter", "std",     0.01)
        self.use_dropout         = _get("dropout",       "enabled", False)
        self.dropout_rate        = _get("dropout",       "rate",    0.05)
        self.use_scale           = _get("scale",         "enabled", False)
        self.scale_range         = tuple(_get("scale",   "range",   [0.95, 1.05]))
        self.use_rotation_y      = _get("rotation_y",   "enabled", False)

    def needs_second_sample(self) -> bool:
        return self.use_cutmix

    def apply(self, sample: dict, second_sample: dict = None) -> dict:
        """
        Apply all enabled augmentations to *sample*.

        Args:
            sample:        Primary sample dict (modified out-of-place).
            second_sample: Required when CutMix is enabled; ignored otherwise.

        Returns:
            Augmented sample dict.
        """
        # ── CutMix ────────────────────────────────────────────────────────────
        if self.use_cutmix and second_sample is not None:
            if np.random.rand() < self.cutmix_prob:
                sample = spatial_cutmix(sample, second_sample, self.design_parameters)

        pcd     = sample["garment_pcd"]
        fps     = sample["garment_fps"]
        q       = sample["q"]
        udf     = sample["udf"]
        has_bm  = "boxmesh_pcd" in sample
        bm_pcd  = sample["boxmesh_pcd"] if has_bm else None
        bm_fps  = sample["boxmesh_fps"] if has_bm else None

        # ── Isotropic scale (applied consistently to garment + boxmesh) ───────
        if self.use_scale:
            pcd, fps, q, udf = _pcd_random_scale(pcd, fps, q, udf, self.scale_range)
            if has_bm:
                # reuse same scale; bm_pcd has no UDF, use dummy tensors
                bm_pcd, bm_fps, _, _ = _pcd_random_scale(
                    bm_pcd, bm_fps,
                    torch.empty(0), torch.empty(0),
                    self.scale_range,
                )

        # ── Rotation around Y (consistent across garment + boxmesh + queries) ─
        if self.use_rotation_y:
            pcd, fps, q = _pcd_random_rotation_y(pcd, fps, q)
            if has_bm:
                # reuse a fresh rotation (independent; garments need not align)
                bm_pcd, bm_fps, _ = _pcd_random_rotation_y(
                    bm_pcd, bm_fps, torch.empty(0, 3)
                )

        # ── XYZ noise ─────────────────────────────────────────────────────────
        if self.use_xyz_noise:
            pcd = _pcd_add_noise(pcd, self.xyz_noise_std)
            fps = _pcd_add_noise(fps, self.xyz_noise_std)
            if has_bm:
                bm_pcd = _pcd_add_noise(bm_pcd, self.xyz_noise_std)
                bm_fps = _pcd_add_noise(bm_fps, self.xyz_noise_std)

        # ── Normal jitter ─────────────────────────────────────────────────────
        if self.use_norm_jitter:
            pcd = _pcd_jitter_normals(pcd, self.norm_jitter_std)
            fps = _pcd_jitter_normals(fps, self.norm_jitter_std)
            if has_bm:
                bm_pcd = _pcd_jitter_normals(bm_pcd, self.norm_jitter_std)
                bm_fps = _pcd_jitter_normals(bm_fps, self.norm_jitter_std)

        # ── Point dropout ─────────────────────────────────────────────────────
        if self.use_dropout:
            pcd = _pcd_random_dropout(pcd, pcd.shape[0], self.dropout_rate)
            if has_bm:
                bm_pcd = _pcd_random_dropout(bm_pcd, bm_pcd.shape[0], self.dropout_rate)

        sample = dict(sample)
        sample["garment_pcd"] = pcd
        sample["garment_fps"] = fps
        sample["q"]           = q
        sample["udf"]         = udf
        if has_bm:
            sample["boxmesh_pcd"] = bm_pcd
            sample["boxmesh_fps"] = bm_fps
        return sample
