from pathlib import Path
import torch
import sys

# Add local warp to path to override installed version
sys.path.insert(0, str(Path(__file__).parent / "warp"))

# Add source directory to path
sys.path.append(str(Path(__file__).parent))

# Project specific imports
from src.model.vae.hy3d_aligned import HunyuanMeshAligned
from src.model.vae.stemnet import StemNet
from src.util.design_params_reader import DesignParameters, PanelGroupedAttentionHeads


def _make_box_and_design(device, LATENT_QUERY_NUM, IMPORTANCE_PERCENTAGE,
                         BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH):
    """Box->sim encoder + design-parameter module."""
    box_to_sim_model = HunyuanMeshAligned(
        latent_query_num=LATENT_QUERY_NUM,
        importance_percentage=IMPORTANCE_PERCENTAGE,
        boundary_percentage=BOUNDARY_PERCENTAGE,
        ckpt=None,
    ).to(device)
    design_parameters = DesignParameters(path=DESIGN_PARAMS_PATH).to(device)
    return box_to_sim_model, design_parameters


def _strip_state_dict_keys(state_dict):
    """Strip Lightning-wrapper prefixes so checkpoint weights map onto the bare model."""
    state_dict = {k.replace("multi_vae.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("udf_net.shape_model.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_stemnet(ckpt_path, device, LATENT_QUERY_NUM, IMPORTANCE_PERCENTAGE, BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH):
    """Build StemNet (PanelGroupedAttentionHeads) and load checkpoint weights."""
    print(f"Loading StemNet from {ckpt_path}")

    box_to_sim_model, design_parameters = _make_box_and_design(
        device, LATENT_QUERY_NUM, IMPORTANCE_PERCENTAGE,
        BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH)

    sim_to_pattern_model = PanelGroupedAttentionHeads(
        design_params=design_parameters,
        input_dim=1024,
        dropout=0.1,
        head_version="v3",
        num_attention_heads=8,
    ).to(device)

    model = StemNet(
        design_parameters=design_parameters,
        box_to_sim_model=box_to_sim_model,
        sim_to_pattern_model=sim_to_pattern_model,
        use_discriminative_latents=True,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _strip_state_dict_keys(checkpoint['state_dict'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print("Weights loaded successfully.")
    model.eval()
    return model, design_parameters


def load_model(ckpt_path, device, LATENT_QUERY_NUM, IMPORTANCE_PERCENTAGE, BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH):
    """Load a StemNet checkpoint.

    Only the 'final' architecture (StemNet + PanelGroupedAttentionHeads) is
    supported in this repo; the legacy v1/v2 pattern predictors were removed. Passing an
    old v1/v2 checkpoint will surface as missing/unexpected state-dict keys.
    """
    return load_stemnet(ckpt_path, device, LATENT_QUERY_NUM, IMPORTANCE_PERCENTAGE, BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH)
