"""
UDF precompute: turn simulated garments into the per-garment training NPZ.

For every garment that has a `<name>_sim.{obj,ply}` and a `<name>_boxmesh.{obj,ply}`
next to each other, this writes `<name>_udf_data_0.01.npz` (keys: garment_pcd,
boxmesh_pcd, q, udf) in the same folder — the exact file the VAE dataloaders
(`gcd_all.py`, `chatgarment.py`) read.

It discovers garments by globbing for `*_sim.{obj,ply}` under --root, so it works
for both the flat GCD layout
    <root>/garments_5000_X/default_body/data/<name>/<name>_sim.ply
and the nested ChatGarment new_garments layout
    <root>/.../<name>/<name>/<name>_sim.obj
including the random-body trees (just point --root at them too).

Meshes are normalized with the shared body-frame `Normalizer` (fixed scale +
translation), so an upper and a lower simulated on the same body stay in their
correct relative position when their point clouds are later concatenated.

Run sharded across an array job:
    python scripts/prepare_data.py --root <dir> --div 16 --part ${SLURM_ARRAY_TASK_ID} --gpu 0
"""
import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import trimesh
import torch
import tqdm

mp.set_start_method("spawn", force=True)

import kaolin as kal
from kaolin.ops.mesh import index_vertices_by_faces

from src.util.udf_sampling import Normalizer

# Match the original GCD precompute budgets (gcd_all.py subsamples from these).
PCD_SIZE  = 10240 * 2     # surface points sampled per garment (split across pieces)
QUERY_NUM = 5120 * 2      # UDF query points
SEED      = 42

normalizer = Normalizer()


# ── geometry helpers (verbatim behaviour from the original prepare_data.py) ────
def sample_corresponding_points(mesh1, mesh2, points, face_indices):
    bary_coords = []
    for p, f in zip(points, face_indices):
        tri = mesh1.vertices[mesh1.faces[f]]
        v0, v1, v2 = tri
        v0v1, v0v2, v0p = v1 - v0, v2 - v0, p - v0
        d00 = np.dot(v0v1, v0v1); d01 = np.dot(v0v1, v0v2); d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1);  d21 = np.dot(v0p, v0v2)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary_coords.append([u, v, w])
    bary_coords = np.array(bary_coords)

    tri2 = mesh2.vertices[mesh2.faces[face_indices]]
    corresponding_points = (
        bary_coords[:, 0:1] * tri2[:, 0, :] +
        bary_coords[:, 1:2] * tri2[:, 1, :] +
        bary_coords[:, 2:3] * tri2[:, 2, :]
    )

    vnorms1 = mesh1.vertex_normals[mesh1.faces[face_indices]]
    normals1 = (bary_coords[:, 0:1] * vnorms1[:, 0, :] +
                bary_coords[:, 1:2] * vnorms1[:, 1, :] +
                bary_coords[:, 2:3] * vnorms1[:, 2, :])
    normals1 = normals1 / (np.linalg.norm(normals1, axis=1, keepdims=True) + 1e-9)
    return points, normals1, corresponding_points, None


def sample_pts_from_meshes(sim_mesh, box_mesh, sfc_pts_num):
    srf_pts, face_idxs = sim_mesh.sample(sfc_pts_num, return_index=True)
    points, normals, corr_points, _ = sample_corresponding_points(
        sim_mesh, box_mesh, points=srf_pts, face_indices=face_idxs)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    extra = np.zeros((sfc_pts_num, 1))
    # boxmesh point copies the sim normal (copy_normals=True in the original)
    src_pcd  = np.concatenate([points, normals, extra], axis=1)
    corr_pcd = np.concatenate([corr_points, normals, extra], axis=1)
    return src_pcd, corr_pcd


@torch.no_grad()
def get_proximity_query_fast(sim_mesh, num_samples, device):
    vertices = torch.from_numpy(sim_mesh.vertices).float().to(device)
    faces = torch.from_numpy(sim_mesh.faces).long().to(device)
    pts, _ = kal.ops.mesh.sample_points(vertices.unsqueeze(0), faces, num_samples)
    pts = pts.squeeze(0)
    queries = torch.cat([
        pts + (torch.rand_like(pts) * 0.003 - 0.0015),
        pts + (torch.rand_like(pts) * 0.003 - 0.0015),
        pts + (torch.rand_like(pts) * 0.01  - 0.005),
        pts + (torch.rand_like(pts) * 0.01  - 0.005),
        pts + (torch.rand_like(pts) * 0.05  - 0.025),
        pts + (torch.rand_like(pts) * 0.1   - 0.05),
        torch.rand_like(pts) * 2 - 1,
    ], dim=0)
    face_vertices = index_vertices_by_faces(vertices.unsqueeze(0), faces)
    dist2, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
        queries.unsqueeze(0), face_vertices)
    udf = torch.sqrt(dist2.squeeze(0))
    return queries.cpu().numpy(), udf.cpu().numpy()


# ── discovery ─────────────────────────────────────────────────────────────────
def _sibling(sim_path, suffix):
    """`/a/b/<name>_sim.obj` -> `/a/b/<name>_boxmesh.obj` (or .ply)."""
    for ext in (".obj", ".ply"):
        cand = sim_path.replace("_sim.obj", f"_{suffix}{ext}").replace("_sim.ply", f"_{suffix}{ext}")
        if os.path.exists(cand):
            return cand
    return None


def build_sim_list(root):
    sims = sorted(glob.glob(os.path.join(root, "**", "*_sim.obj"), recursive=True) +
                  glob.glob(os.path.join(root, "**", "*_sim.ply"), recursive=True))
    todo = []
    skipped = missing_box = 0
    for s in sims:
        folder = os.path.dirname(s)
        name = os.path.basename(s).replace("_sim.obj", "").replace("_sim.ply", "")
        out = os.path.join(folder, f"{name}_udf_data_0.01.npz")
        if os.path.exists(out):
            try:
                with np.load(out) as d:
                    _ = d["garment_pcd"], d["boxmesh_pcd"], d["q"], d["udf"]
                skipped += 1
                continue
            except Exception:
                pass  # corrupted -> redo
        if _sibling(s, "boxmesh") is None:
            missing_box += 1
            continue
        todo.append(s)
    print(f"Found {len(sims)} sims | todo={len(todo)} | already-done={skipped} | missing-boxmesh={missing_box}")
    return todo


def preprocess_one(sim_path, device):
    folder = os.path.dirname(sim_path)
    name = os.path.basename(sim_path).replace("_sim.obj", "").replace("_sim.ply", "")
    out = os.path.join(folder, f"{name}_udf_data_0.01.npz")
    try:
        box_path = _sibling(sim_path, "boxmesh")
        sim_mesh = normalizer.normalize(trimesh.load(sim_path, process=False))
        box_mesh = normalizer.normalize(trimesh.load(box_path, process=False))
        garment_pcd, boxmesh_pcd = sample_pts_from_meshes(sim_mesh, box_mesh, PCD_SIZE)
        q, udf = get_proximity_query_fast(sim_mesh, QUERY_NUM, device=device)
        np.savez(
            out,
            garment_pcd=garment_pcd.astype(np.float32),
            boxmesh_pcd=boxmesh_pcd.astype(np.float32),
            q=q.astype(np.float32),
            udf=udf.astype(np.float32),
        )
    except Exception as e:
        print(f"[FAILED] {name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir to glob for *_sim.{obj,ply}")
    ap.add_argument("--div", type=int, default=1, help="Total shards")
    ap.add_argument("--part", type=int, default=0, help="This shard index [0, div)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(SEED)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    sims = build_sim_list(args.root)
    n = len(sims)
    start, end = args.part * n // args.div, (args.part + 1) * n // args.div
    shard = sims[start:end]
    print(f"Shard {args.part+1}/{args.div}: {len(shard)} garments on {device}")

    for s in tqdm.tqdm(shard):
        preprocess_one(s, device)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
