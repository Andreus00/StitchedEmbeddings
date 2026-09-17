"""
inference_common.py
--------------------
Shared helpers for the inference / demo entry scripts (run_tto_delta_on_test_set.py,
demo_edit_garment_with_predicted_normals.py, test_normal_flow.py, test_garment_dit.py).

This module only collects code that was byte-for-byte duplicated across those scripts.
Values that intentionally differ between scripts (e.g. MAX_DIST 0.01 vs 0.0098, the body
path) are deliberately left local to each script and are NOT defined here.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants shared (identical) across run_tto_delta and demo_edit
# ---------------------------------------------------------------------------
LATENT_QUERY_NUM      = 256
PCD_PTS_NUM           = LATENT_QUERY_NUM * 20   # 5120
IMPORTANCE_PERCENTAGE = 0.
BOUNDARY_PERCENTAGE   = 0.
DESIGN_PARAMS_PATH    = "src/external/GarmentCodeAssets/assets/design_params/default.yaml"


# ---------------------------------------------------------------------------
# Flow-matching ODE integration for normal prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def ode_integrate_normals(flow_model, pts, num_steps, seed=None, method="euler"):
    """
    Integrate the learned velocity field from z0 ~ N(0,I) at t=0 to z1 at t=1.

    Args:
        flow_model: FlowNormalTrainer (or compatible velocity field f(pts, z, t)).
        pts:        (B, N, 3) batch of point clouds.
        num_steps:  number of integration steps.
        seed:       optional RNG seed for reproducible z0 sampling.
        method:     "euler" (default) or "heun" (2nd-order, 2 model evals/step).

    Returns:
        (B, N, 3) unit-normalized predicted normals.
    """
    B, N, _ = pts.shape
    device = pts.device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    z = torch.randn(B, N, 3, device=device, generator=gen)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t0 = torch.full((B,), step * dt, device=device)
        v0 = flow_model(pts, z, t0)
        if method == "heun":
            t1 = torch.full((B,), (step + 1) * dt, device=device)
            v1 = flow_model(pts, z + dt * v0, t1)
            z = z + dt * 0.5 * (v0 + v1)
        else:
            z = z + dt * v0
    return F.normalize(z, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Normal-model loading (try flow-matching, fall back to MLP/GNN autodecoder)
# ---------------------------------------------------------------------------
def load_normal_model(ckpt_path, device):
    """Load a normal predictor as FlowNormalTrainer, falling back to NormalPredictorTrainer."""
    # Imported lazily to avoid pulling the trainer stack into scripts that don't need it.
    from src.model.normal_flow_trainer import FlowNormalTrainer
    from src.model.normal_trainer import NormalPredictorTrainer
    try:
        model = FlowNormalTrainer.load_from_checkpoint(ckpt_path, map_location=device)
        print("  [normal model] loaded as FlowNormalTrainer")
    except Exception:
        model = NormalPredictorTrainer.load_from_checkpoint(ckpt_path, map_location=device)
        print("  [normal model] loaded as NormalPredictorTrainer")
    model.eval().to(device)
    return model
