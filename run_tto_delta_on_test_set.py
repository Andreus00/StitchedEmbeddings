"""
Test-time optimization (delta variant) on the GarmentCode test set.
"""

import argparse
import os
import json
import shutil
from pathlib import Path
import torch
from tqdm import tqdm
import trimesh
import sys

import kaolin as kal
from kaolin.ops.mesh import index_vertices_by_faces
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.util.udf_sampling import Normalizer
from src.external.GarmentCode.pattern_sampler import _save_sample
from src.external.GarmentCode.assets.garment_programs.meta_garment import MetaGarment
from src.external.GarmentCode.assets.bodies.body_params import BodyParameters
from models import load_model
from evaluate_specification import evaluate_prediction
from src.util.inference_common import (
    LATENT_QUERY_NUM, PCD_PTS_NUM, IMPORTANCE_PERCENTAGE, BOUNDARY_PERCENTAGE,
    DESIGN_PARAMS_PATH,
)

MAX_DISTANCE      = 0.01
DEFAULT_CKPT      = "checkpoints/best-checkpoint-v26.ckpt"
DEFAULT_BODY_PATH = "src/external/GarmentCode/assets/bodies/mean_all.yaml"


def get_args():
    parser = argparse.ArgumentParser(description='Delta TTO on GarmentCode test set')
    parser.add_argument('--data_root',   type=str, default='data/sample/GarmentCodeData_v2/')
    parser.add_argument('--split_file',  type=str, default='data/sample/split.json')
    parser.add_argument('--output',      type=str, default='results_tto_delta_testset')
    parser.add_argument('--ckpt',        type=str, default=DEFAULT_CKPT)
    parser.add_argument('--device',      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_steps',   type=int, default=1000)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--max_folders', type=int, default=None)
    parser.add_argument('--delta_reg_weight', type=float, default=1.0,
                        help='L2 penalty on delta (keeps latent near encoder anchor)')
    parser.add_argument('--kl_reg_weight',    type=float, default=1e-3,
                        help='L2 penalty on perturbed latent (keeps near N(0,I) prior)')
    parser.add_argument('--num_query_points', type=int, default=2048)
    parser.add_argument('--eval_every',  type=int, default=0,
                        help='Evaluate metrics every N steps (0 = only at end)')
    return parser.parse_args()


@torch.no_grad()
def get_proximity_query_fast(sim_mesh: trimesh.Trimesh, num_samples: int, device="cuda"):
    vertices = torch.from_numpy(sim_mesh.vertices).float().to(device)
    faces    = torch.from_numpy(sim_mesh.faces).long().to(device)
    pts, _   = kal.ops.mesh.sample_points(vertices.unsqueeze(0), faces, num_samples)
    pts      = pts.squeeze(0)
    queries  = torch.cat([
        pts + (torch.rand_like(pts) * 0.003 - 0.0015),
        pts + (torch.rand_like(pts) * 0.003 - 0.0015),
        pts + (torch.rand_like(pts) * 0.01  - 0.005),
        pts + (torch.rand_like(pts) * 0.01  - 0.005),
        pts + (torch.rand_like(pts) * 0.05  - 0.025),
        pts + (torch.rand_like(pts) * 0.05  - 0.025),
        pts + (torch.rand_like(pts) * 0.1   - 0.05),
        torch.rand_like(pts) * 2 - 1,
        torch.rand_like(pts) * 2 - 1,
        torch.rand_like(pts) * 2 - 1,
        torch.rand_like(pts) * 2 - 1,
        torch.rand_like(pts) * 2 - 1,
    ], dim=0)
    face_vertices = index_vertices_by_faces(vertices.unsqueeze(0), faces)
    dist2, _, _   = kal.metrics.trianglemesh.point_to_mesh_distance(queries.unsqueeze(0), face_vertices)
    udf = torch.sqrt(dist2.squeeze(0))
    return queries.cpu().numpy(), udf.cpu().numpy()


def prepare_queries(mesh, num_samples, device, dtype):
    q, gt_udf = get_proximity_query_fast(mesh, num_samples, device=device)
    q      = torch.from_numpy(q).to(device).to(dtype).unsqueeze(0)
    gt_udf = torch.clamp(torch.from_numpy(gt_udf).to(device).to(dtype), max=MAX_DISTANCE) / MAX_DISTANCE
    return q, gt_udf


def load_mesh(mesh_path, normalizer, device, dtype):
    try:
        sim_mesh     = trimesh.load(mesh_path, process=False)
        sim_mesh     = normalizer.normalize(sim_mesh)
        sim_pcd, sim_tri = trimesh.sample.sample_surface(sim_mesh, PCD_PTS_NUM)
        sim_pcd      = torch.from_numpy(sim_pcd).to(device).to(dtype)
        sim_normals  = torch.from_numpy(sim_mesh.face_normals[sim_tri]).to(device).to(dtype)
        sim_features = torch.zeros((sim_pcd.shape[0], 1), device=device, dtype=dtype)
        sim_input    = torch.cat([sim_pcd, sim_normals, sim_features], dim=1)
        return sim_mesh, sim_input
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None, None


def save_prediction(specs, out_name, out_path, default_body):
    try:
        dm = MetaGarment(out_name, default_body, specs[0]['design'])
        _save_sample(dm, default_body, specs[0]['design'], out_path, verbose=False)
    except Exception as e:
        print(f"Error saving prediction {out_name}: {e}")


def _load_spec(path):
    with open(path) as f:
        return json.load(f)


def process_single(subdir, model, design_params, normalizer, args, default_body):
    device = args.device
    dtype  = torch.float32

    # Version-aware API selection (same pattern as 4ddress script)
    is_v2       = hasattr(model, 'garment_encoder')
    encoder     = model.garment_encoder if is_v2 else model.pcd_encoder
    classify_fn = model.classify_latents if is_v2 else model.classify
    decode_udf  = model.decode3d         if is_v2 else model.decode

    basename        = os.path.basename(subdir)
    mesh_path       = os.path.join(subdir, basename + '_sim.ply')
    gt_pattern_path = os.path.join(subdir, basename + '_specification.json')
    gt_design_path  = os.path.join(subdir, basename + '_design_params.yaml')
    gt_svg_path     = os.path.join(subdir, basename + '_pattern.svg')
    garment_id      = '/'.join(str(mesh_path).split('/')[-4:-1])
    out_dir         = Path(args.output) / garment_id

    if out_dir.exists():
        print(f"Skipping {garment_id}, already exists.")
        return None

    sim_mesh, sim_input = load_mesh(mesh_path, normalizer, device, dtype)
    if sim_mesh is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(gt_pattern_path, out_dir / "gt_pattern.json")
    shutil.copy(gt_design_path,  out_dir / "gt_design.yaml")
    shutil.copy(gt_svg_path,     out_dir / "gt_pattern.svg")
    shutil.copy(mesh_path,       out_dir / "sim_gt.ply")
    print(f"Processing {garment_id}")

    sim_input_batch = sim_input.unsqueeze(0)  # [1, N, 7]
    ground_truth    = _load_spec(gt_pattern_path)
    step_metrics    = []

    # -- 1. Initial inference (deterministic KL mean) --
    with torch.no_grad():
        moments = encoder(sim_input_batch)
        _, latents = model.run_kl(moments, deterministic=True)

        predicted_params = classify_fn(latents)["predicted_parameters"]
        all_specs = design_params._prediction_to_multiple_yaml(predicted_params, differentiable=False)
        save_prediction([all_specs[0]], "initial_prediction", str(out_dir), default_body)

        init_path = out_dir / "initial_prediction/initial_prediction_specification.json"
        init_metrics = evaluate_prediction(_load_spec(init_path), ground_truth)
        init_metrics['phase'] = 'initial'
        init_metrics['step']  = -1
        step_metrics.append(init_metrics)

    print(f"  initial: matched_panels={init_metrics['num_matched_panels']}/{init_metrics['num_panels_gt']}")

    # -- 2. Delta optimization --
    anchor_latent = latents.clone().detach()
    delta = torch.zeros_like(anchor_latent, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    # Fixed query set — one seed, computed once for stable gradients
    torch.manual_seed(42)
    q, gt_udf = prepare_queries(sim_mesh, args.num_query_points, device, dtype)

    delta_reg_weight = args.delta_reg_weight
    kl_reg_weight    = args.kl_reg_weight

    for i in range(args.max_steps):
        optimizer.zero_grad()

        perturbed   = anchor_latent + delta
        decoded_udf = decode_udf(perturbed, q).squeeze(-1).flatten()
        udf_loss    = torch.nn.functional.mse_loss(decoded_udf, gt_udf)
        delta_reg   = torch.mean(delta ** 2)
        kl_reg      = 0.5 * torch.mean(perturbed ** 2)
        loss        = udf_loss + delta_reg_weight * delta_reg + kl_reg_weight * kl_reg

        loss.backward()
        optimizer.step()

        row = {
            'phase':      'opt',
            'step':       i,
            'loss':       loss.item(),
            'udf_loss':   udf_loss.item(),
            'delta_norm': delta.detach().norm().item(),
            'kl_reg':     kl_reg.item(),
        }

        if args.eval_every > 0 and i > 0 and i % args.eval_every == 0:
            with torch.no_grad():
                p_params   = classify_fn(anchor_latent + delta.detach())["predicted_parameters"]
                specs_i    = design_params._prediction_to_multiple_yaml(p_params, differentiable=False)
                save_prediction([specs_i[0]], f"step_{i}", str(out_dir), default_body)
                inter_path = out_dir / f"step_{i}/step_{i}_specification.json"
                m = evaluate_prediction(_load_spec(inter_path), ground_truth)
                row.update({f'eval_{k}': v for k, v in m.items() if isinstance(v, (int, float, bool))})

        step_metrics.append(row)

        if i % 100 == 0:
            print(f"  step {i:4d}: udf={udf_loss.item():.5f} delta_norm={delta.detach().norm().item():.3f}")

    # -- 3. Final prediction --
    with torch.no_grad():
        final_latent     = anchor_latent + delta.detach()
        predicted_params = classify_fn(final_latent)["predicted_parameters"]
        all_specs_f      = design_params._prediction_to_multiple_yaml(predicted_params, differentiable=False)
        save_prediction([all_specs_f[0]], "final_prediction", str(out_dir), default_body)

        final_path    = out_dir / "final_prediction/final_prediction_specification.json"
        final_metrics = evaluate_prediction(_load_spec(final_path), ground_truth)
        final_metrics['phase'] = 'final'
        final_metrics['step']  = args.max_steps
        step_metrics.append(final_metrics)

    print(f"  final:   matched_panels={final_metrics['num_matched_panels']}/{final_metrics['num_panels_gt']}")

    pd.DataFrame(step_metrics).to_csv(out_dir / "optimization_metrics.csv", index=False)
    return {'initial': init_metrics, 'final': final_metrics}


def _aggregate(all_results, phase):
    import numpy as np
    rows = [r[phase] for r in all_results if r is not None and phase in r]
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if np.isscalar(v) and not isinstance(v, str)]
    return {
        k: float(np.mean([r[k] for r in rows if k in r and np.isfinite(float(r[k]))]))
        for k in keys
    }


def main():
    args = get_args()

    if not os.path.exists(args.data_root):
        print(f"Data root {args.data_root} does not exist.")
        return

    model, design_params = load_model(
        args.ckpt, args.device, LATENT_QUERY_NUM,
        IMPORTANCE_PERCENTAGE, BOUNDARY_PERCENTAGE, DESIGN_PARAMS_PATH
    )
    if model is None:
        print("Failed to load model.")
        return

    default_body = BodyParameters(Path(DEFAULT_BODY_PATH))
    normalizer   = Normalizer()

    with open(args.split_file, 'r') as f:
        train_val_test = json.load(f)
    subdirs = [
        os.path.join(args.data_root, p.replace('default_body', 'default_body/data'))
        for p in train_val_test['test']
    ]
    subdirs.sort()
    if args.max_folders:
        subdirs = subdirs[:args.max_folders]
    print(f"Found {len(subdirs)} test garments.")

    all_results = []
    for subdir in tqdm(subdirs, desc="Processing"):
        result = process_single(subdir, model, design_params, normalizer, args, default_body)
        all_results.append(result)

    import numpy as np

    # -- Aggregate and print summary --
    init_agg  = _aggregate(all_results, 'initial')
    final_agg = _aggregate(all_results, 'final')

    METRICS_TO_PRINT = [
        ('num_panels_correct',     '%panels correct',       '{:.1%}'),
        ('avg_vertex_l2',          'avg vertex L2 (cm)',    '{:.4f}'),
        ('avg_translation_l2',     'avg translation L2',    '{:.4f}'),
        ('avg_rotation_l2',        'avg rotation L2 (rad)', '{:.4f}'),
        ('avg_num_edges_accuracy', '%edges accuracy',       '{:.1%}'),
        ('stitch_precision',       'stitch precision',      '{:.1%}'),
        ('stitch_recall',          'stitch recall',         '{:.1%}'),
    ]

    print("\n========== AGGREGATE RESULTS ==========")
    print(f"{'Metric':<28} {'Initial':>10} {'Final':>10} {'Delta':>10}")
    print("-" * 62)
    for key, label, fmt in METRICS_TO_PRINT:
        i_val = init_agg.get(key, float('nan'))
        f_val = final_agg.get(key, float('nan'))
        d_val = f_val - i_val
        sign  = '+' if d_val >= 0 else ''
        print(f"{label:<28} {fmt.format(i_val):>10} {fmt.format(f_val):>10} {sign}{fmt.format(d_val):>9}")

    # Per-sample improvement ranked by vertex L2
    improvements = [
        (r['initial']['avg_vertex_l2'] - r['final']['avg_vertex_l2'],
         r['initial']['avg_vertex_l2'], r['final']['avg_vertex_l2'])
        for r in all_results if r is not None
        and np.isfinite(r['initial'].get('avg_vertex_l2', float('nan')))
        and np.isfinite(r['final'].get('avg_vertex_l2',   float('nan')))
    ]
    improvements.sort(reverse=True)
    N = min(5, len(improvements))
    if improvements:
        print(f"\n  Top {N} most improved (vertex L2):")
        for d, i_v, f_v in improvements[:N]:
            print(f"    Δ={d:+.4f}  init={i_v:.4f} → final={f_v:.4f}")
        print(f"\n  Top {N} most degraded (vertex L2):")
        for d, i_v, f_v in improvements[-N:]:
            print(f"    Δ={d:+.4f}  init={i_v:.4f} → final={f_v:.4f}")

    summary = {
        'initial': init_agg,
        'final':   final_agg,
        'delta':   {k: final_agg.get(k, 0) - init_agg.get(k, 0) for k in init_agg},
    }
    summary_path = Path(args.output) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
