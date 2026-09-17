import numpy as np
import os
import copy
import glob as _glob
import yaml as _yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import tqdm
import trimesh
from src.util.utils import shuffle_list
from enum import Enum
from typing import List
from torch_cluster import fps as fps_fn
from src.util.design_params_reader import DesignParameters
from src.datamodule.datasets.augmentation_gcd import GCDAugmentation
from scipy.spatial import cKDTree
import time
import json

class DataType(Enum):
    LATENT_SHAPE = 0
    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))
  

def print_vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
    
import kaolin as kal
from kaolin.ops.mesh import index_vertices_by_faces

def udf(vertices, faces, x):
    # kaolin expects points: (N, 3), face_vertices: (F, 3, 3)
    face_vertices = index_vertices_by_faces(vertices.unsqueeze(0), faces)  # (F, 3, 3)
    x_batched = x.unsqueeze(0)  # (B, N, 3)
    sq_dist, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
        x_batched,  # (1, N, 3)
        face_vertices  # (1, F, 3, 3)
    )
    return sq_dist.squeeze(0).sqrt()

def sample_mesh_surface(vertices, faces, num_samples):
    # vertices: [B, V, 3]
    # faces: [F, 3]
    # num_samples: int
    pts, face_idx = kal.ops.mesh.sample_points(vertices, faces, num_samples)
    return pts, face_idx

def get_random_queries(vertices, faces, num_samples):
    # vertices: [V, 3]
    # faces: [F, 3]
    # num_samples: int
    pts, face_idx = sample_mesh_surface(vertices.unsqueeze(0), faces, num_samples)

    queries = torch.cat([
                pts + (torch.rand_like(pts, device=vertices.device) * 0.003 - 0.0015),
                pts + (torch.rand_like(pts, device=vertices.device) * 0.01 - 0.005),
                pts + (torch.rand_like(pts, device=vertices.device) * 0.05 - 0.025),
                pts + (torch.rand_like(pts, device=vertices.device) * 0.1 - 0.05),
                torch.rand_like(pts, device=vertices.device) * 2 - 1,
                ], dim=1).squeeze(0)
    
    gt_udf = udf(vertices, faces, queries)

    return queries, gt_udf

def get_proximity_query(mesh: trimesh.Trimesh, num_samples: int):
    # vertices: [V, 3]
    # faces: [F, 3]
    # num_samples: int
    pts = mesh.sample(num_samples)

    q = np.concatenate([
                pts + (np.random.rand(num_samples, 3) * 0.003 - 0.0015),
                pts + (np.random.rand(num_samples, 3) * 0.01 - 0.005),
                pts + (np.random.rand(num_samples, 3) * 0.05 - 0.025),
                pts + (np.random.rand(num_samples, 3) * 0.1 - 0.05),
                np.random.rand(num_samples, 3) * 2 - 1,
                ], axis=0)

    closest, distance, triangle_idx = trimesh.proximity.closest_point(mesh, q)

    return q, distance

def sample_corresponding_points(mesh1, mesh2, points, face_indices):
    # Compute barycentric coordinates for sampled points
    bary_coords = []
    for p, f in zip(points, face_indices):
        tri = mesh1.vertices[mesh1.faces[f]]
        v0, v1, v2 = tri
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0

        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary_coords.append([u, v, w])

    bary_coords = np.array(bary_coords)

    # Get corresponding points in mesh2 using same faces + barycentric coords
    tri2 = mesh2.vertices[mesh2.faces[face_indices]]

    corresponding_points = (
        bary_coords[:, 0:1] * tri2[:, 0, :] +
        bary_coords[:, 1:1+1] * tri2[:, 1, :] +
        bary_coords[:, 2:3] * tri2[:, 2, :]
    )

    # Interpolate per-vertex normals to the sampled points using the same barycentric coords
    vnorms1 = mesh1.vertex_normals[mesh1.faces[face_indices]]  # (n_points, 3 verts, 3)
    normals1 = (
        bary_coords[:, 0:1] * vnorms1[:, 0, :] +
        bary_coords[:, 1:2] * vnorms1[:, 1, :] +
        bary_coords[:, 2:3] * vnorms1[:, 2, :]
    )
    norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
    normals1 = normals1 / (norms1 + 1e-9)

    vnorms2 = mesh2.vertex_normals[mesh2.faces[face_indices]]
    normals2 = (
        bary_coords[:, 0:1] * vnorms2[:, 0, :] +
        bary_coords[:, 1:2] * vnorms2[:, 1, :] +
        bary_coords[:, 2:3] * vnorms2[:, 2, :]
    )
    norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
    normals2 = normals2 / (norms2 + 1e-9)

    return points, normals1, corresponding_points, normals2


def sample_pts_from_meshes(mesh_trimesh: trimesh.Trimesh, mesh_trimesh_2: trimesh.Trimesh, sfc_pts_num: int, copy_normals: bool):
    # sample points from the first surface
    srf_pts, face_idxs = mesh_trimesh.sample(sfc_pts_num, return_index=True)
    # get correspoinding points from the second mesh
    points, normals, corresponding_points, corresponding_normals = sample_corresponding_points(mesh_trimesh, mesh_trimesh_2, points=srf_pts, face_indices=face_idxs)

    # normalize normals to unit length (safe against zero-length)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-9)

    if copy_normals:
        corresponding_normals = normals
    else:
        norm_corresponding_normals = np.linalg.norm(corresponding_normals, axis=1, keepdims=True)
        corresponding_normals = corresponding_normals / (norm_corresponding_normals + 1e-9)

    # concatenate points and features
    additional_feature = np.zeros((sfc_pts_num, 1))
    src_pcd = np.concatenate([points, normals, additional_feature], axis=1)
    corresponding_pcd = np.concatenate([corresponding_points, corresponding_normals, additional_feature], axis=1)

    return src_pcd, corresponding_pcd


def extract_gt_design_params_with_mask(gt_design, design_params_template):
    # gt_design may be a path (str/Path) or an already-built DesignParameters
    # (e.g. a merged top+bottom GT).
    dp = gt_design if isinstance(gt_design, DesignParameters) else DesignParameters(gt_design)

    predictions = {}
    for panel_name, panel_params in dp.panel_data.items():
        predictions[panel_name] = {}
        for param_name, param in panel_params.items():
            predictions[panel_name][param_name] = param.get_value()
            
    active_mask = dp.get_active_mask(predictions)
    
    flat_vector = []
    # Use deterministic order based on sorted panel and parameter names from the template
    sorted_panels = sorted(design_params_template.panel_data.keys())
    for panel_name in sorted_panels:
        panel_params = design_params_template.panel_data[panel_name]
        sorted_params = sorted(panel_params.keys())
        for param_name in sorted_params:
            template_param = panel_params[param_name]
            
            is_active = active_mask.get(panel_name, {}).get(param_name, False)
            if is_active:
                val = template_param.value_of(dp.panel_data[panel_name][param_name])
                if isinstance(val, torch.Tensor):
                    arr = val.flatten().cpu().numpy()
                else:
                    arr = np.array([val]).flatten()
                
                out_size = template_param.get_output_size()
                if arr.size > out_size:
                    arr = arr[:out_size]
                elif arr.size < out_size:
                    arr = np.pad(arr, (0, out_size - arr.size))
                flat_vector.append(arr)
            else:
                out_size = template_param.get_output_size()
                flat_vector.append(np.zeros(out_size, dtype=np.float32))
                
    if flat_vector:
        return np.concatenate(flat_vector, axis=0).astype(np.float32)
    return np.array([], dtype=np.float32)


# Panels that belong to the upper vs the lower garment — used to merge a separate
# top + bottom into one combined design for two-piece outfits.
_UPPER_PANEL_KEYS = ['shirt', 'collar', 'sleeve', 'left']
_BOTTOM_PANEL_KEYS = ['waistband', 'skirt', 'flare-skirt', 'godet-skirt',
                      'pencil-skirt', 'levels-skirt', 'pants', 'cut']


def _resolve_npz(folder):
    # prepare_data.py writes the npz next to the sim mesh, which the pool /
    # pair layouts nest one level deeper (<folder>/<name>/<name>_sim.*)
    hits = (_glob.glob(os.path.join(folder, '*_udf_data_0.01.npz'))
            or _glob.glob(os.path.join(folder, '**', '*_udf_data_0.01.npz'), recursive=True))
    return hits[0] if hits else os.path.join(folder, f"{os.path.basename(folder)}_udf_data_0.01.npz")


def _resolve_design_yaml(folder):
    name = os.path.basename(folder)
    for cand in (f"{name}_design_params.yaml", "design_params.yaml", "design.yaml"):
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            return p
    return None


def _load_design_inner(path):
    return _yaml.safe_load(open(path))['design']


def _merge_top_bottom_designs(top_inner, bottom_inner):
    """Combine an upper-only design (top) and a lower-only design (bottom) into a
    single coherent design dict: upper panels + meta.upper from the top, bottom
    panels + meta.bottom/wb from the bottom.

    Both pieces can carry a waistband. The bottom's stays in the canonical
    meta.wb + design.waistband slots; the top's moves to meta.top_wb +
    design.top_waistband (template groups added for two-piece outfits) so the
    two don't clobber each other."""
    merged = copy.deepcopy(top_inner)
    # Re-home the top's waistband before the bottom's overwrites the shared
    # slots below. meta.wb is the piece's own selector in a standalone design.
    top_wb = (top_inner.get('meta', {}) or {}).get('wb')
    if top_wb is not None and top_wb.get('v') is not None:
        merged.setdefault('meta', {})['top_wb'] = copy.deepcopy(top_wb)
        if 'waistband' in top_inner:
            merged['top_waistband'] = copy.deepcopy(top_inner['waistband'])
    # The shared slot defaults to "no waistband" unless the bottom brings one
    # (keep the full param structure — DesignParameters needs type/range).
    if top_wb is not None and 'wb' in merged.get('meta', {}):
        merged['meta']['wb'] = copy.deepcopy(top_wb)
        merged['meta']['wb']['v'] = None
    for k in _BOTTOM_PANEL_KEYS:
        if k in bottom_inner:
            merged[k] = copy.deepcopy(bottom_inner[k])
    if 'meta' in bottom_inner:
        merged.setdefault('meta', {})
        for mk in ('bottom', 'wb'):
            if mk in bottom_inner['meta']:
                merged['meta'][mk] = copy.deepcopy(bottom_inner['meta'][mk])
    return merged


## Dataset that provides Simulated ad BoxMesh pointcloud
class GCD_ALL(Dataset):
    def __init__(
        self,
        root_dir: str,
        load_by_dataname: List[Tuple[str, str, str]],
        small_dataset: bool,
        device: str,
        seed: int,
        split: str,
        debug: bool,
        num_elements_debug: int,
        pcd_size: int,
        imp_size_percent: float,
        bnd_size_percent: float,
        query_num: int,
        model_normalization_type: str,
        absolute_scale: float,
        sample_from: str,
        design_parameters: DesignParameters,
        train_val_test_file: str,
        augmentation: dict = None,
        ):

        if debug is True and split == 'train':
            split = 'debug'
        if debug is True and (split == 'test' or split == 'validation'):
            split = 'debug_test'

        assert split in ['train', 'validation', 'test', 'debug', 'debug_test', 'debug_test_unseen']

        #################################
        # init from the basedataset class
        self.root_path = Path(root_dir)
        self.split = split
        self.seed = seed
        self.device = device

        self.design_parameters: DesignParameters = design_parameters

        self.sample_from = sample_from

        self.imp_size = int(pcd_size * imp_size_percent)
        self.bnd_size = int(pcd_size * bnd_size_percent)
        self.pcd_size = int(pcd_size - self.imp_size - self.bnd_size)

        self.query_num = query_num

        self.model_normalization_type = model_normalization_type
        self.absolute_scale = absolute_scale

        # New (outfit) format: train_val_test_file is
        #   {split: [{"pieces": [<folder>, ...], "is_two_piece": bool}, ...]}
        # where each folder holds a *_udf_data_0.01.npz. One piece = dress/whole,
        # two pieces = top+bottom. Built by scripts/build_split.py.
        self.new_format = False
        if train_val_test_file is not None and not small_dataset:
            try:
                with open(train_val_test_file) as _f:
                    _probe = json.load(_f)
                _key = {'debug': 'train', 'debug_test': 'test',
                        'debug_test_unseen': 'test'}.get(self.split, self.split)
                if (isinstance(_probe, dict) and _probe.get(_key)
                        and isinstance(_probe[_key][0], dict) and 'pieces' in _probe[_key][0]):
                    self.new_format = True
                    load_by_dataname = list(_probe[_key])
                    if debug:
                        load_by_dataname = load_by_dataname[:num_elements_debug]
            except Exception as _e:
                print(f"new-format probe failed ({_e}); falling back to legacy split logic")

        if self.new_format:
            pass  # load_by_dataname already holds the outfit entries
        elif small_dataset:
            # load only a folder with a few garment names
            folders = sorted(os.listdir(self.root_path))
            if not folders:
                raise FileNotFoundError(f"No folders found in {self.root_path}")
            first_folder = folders[0]
            garments_path = os.path.join(self.root_path, first_folder, "default_body", "data")
            garment_folders = sorted(os.listdir(garments_path))
            load_by_dataname = [os.path.join(garments_path, g) for g in garment_folders if os.path.isdir(os.path.join(garments_path, g))]
            num_samples = len(load_by_dataname)
            frac = num_samples // 10
            split_bounds = {
                'train': (0, frac * 8, 4), # take 80% of the dataset. Repeat 4 times for augmentation
                'validation': (frac * 8, frac * 8 + frac, 1), # take 10% of the dataset
                'test': (frac * 8 + frac, frac * 8 + frac + frac, 1), # take 10% of the dataset
                'debug': (4, 4 + num_elements_debug, 100),#int(1000 / max(1, num_elements_debug//10))),   # for debug take a few samples
                'debug_test': (4, 4 + num_elements_debug, 1),   # for debug take a few samples
                'debug_test_unseen': (4 + num_elements_debug, 4 + num_elements_debug + num_elements_debug, 1)   # for debug take a few samples
            }[self.split]

            print(f"Loading {split_bounds} for split {self.split}")
            load_by_dataname = load_by_dataname[split_bounds[0]:split_bounds[1]] * split_bounds[2]
            
        else:
            if self.split in ['train', 'validation', 'test']:
                with open(train_val_test_file, 'r') as f:
                    train_val_test = json.load(f)
                load_by_dataname = train_val_test[self.split]
                load_by_dataname = [os.path.join(self.root_path, p.replace('default_body', 'default_body/data')) for p in load_by_dataname]
            else:
                folders = sorted(os.listdir(self.root_path))
                if not folders:
                    raise FileNotFoundError(f"No folders found in {self.root_path}")
                load_by_dataname = []
                for part in tqdm.tqdm(folders, total=len(folders), desc="Loading all the paths"):
                    garments_path = os.path.join(self.root_path, part, "default_body", "data")
                    garment_folders = sorted(os.listdir(garments_path))
                    load_by_dataname += [os.path.join(garments_path, g) for g in garment_folders if os.path.isdir(os.path.join(garments_path, g))]
                load_by_dataname = shuffle_list(load_by_dataname, seed)
                num_samples = len(load_by_dataname)
                frac = num_samples // 10
                split_bounds = {
                    'train': (0, frac * 8, 1), # take 80% of the dataset
                    'validation': (frac * 8, frac * 8 + frac, 1), # take 10% of the dataset
                    'test': (frac * 8 + frac, frac * 8 + frac + int(frac*0.1), 1), # take 10% of the dataset
                    # 'test': (frac * 8 + frac, frac * 8 + frac + 8, 1), # take 10% of the dataset
                    'debug': (4, 4 + num_elements_debug, int(1000 / max(1, num_elements_debug//10))),   # for debug take a few samples
                    'debug_test': (4, 4 + num_elements_debug, 10)   # for debug take a few samples
                }[self.split]

                print(f"Loading {split_bounds} for split {self.split}")
                load_by_dataname = load_by_dataname[split_bounds[0]:split_bounds[1]] * split_bounds[2]

        self.all_npz_list = load_by_dataname
        self.datapoints_names = load_by_dataname
        # self.refresh_npz_list()
        print(f"Loaded {len(self.datapoints_names)} datapoints for split {self.split}")

        self.augmentation: GCDAugmentation = None
        if augmentation is not None:
            self.augmentation = GCDAugmentation(augmentation, design_parameters)

    def refresh_npz_list(self):
        self.datapoints_names = [x for x in self.all_npz_list if os.path.exists(os.path.join(x, f"{os.path.basename(x)}_udf_data_0.01.npz"))]

    # added from maria 
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)  

    def get_garment_names(self):
        # New-format entries are dicts ({"pieces": [...], ...}) — serialize to a
        # stable string so the bookkeeping dump in train_stemnet.py works.
        names = ['+'.join(x['pieces']) if isinstance(x, dict) else x
                 for x in self.datapoints_names]
        return list(set(names))

    def _load_outfit(self, entry: dict) -> dict:
        """Load a one-piece (dress) or two-piece (top+bottom) outfit.

        PCDs and UDF queries are split evenly across pieces and concatenated, so a
        two-piece sample is the two un-sewn meshes as one point cloud (they were
        normalized in the shared body frame, so they stay anatomically aligned).
        The UDF target is the per-piece concatenation — an approximation of the
        union field that is accurate where the pieces are spatially disjoint.
        The design GT merges the top's upper panels with the bottom's lower panels;
        meta.is_two_piece carries the dress vs top+bottom label.
        """
        pieces = entry["pieces"]
        is_two = bool(entry.get("is_two_piece", len(pieces) > 1))
        top_tucked = bool(entry.get("top_tucked", False))
        n = len(pieces)
        per_pcd = self.pcd_size // n
        per_q = self.query_num // n

        g_list, b_list, q_list, u_list, design_inners, names = [], [], [], [], [], []
        for folder in pieces:
            names.append(os.path.basename(folder))
            data = np.load(_resolve_npz(folder))
            ip = np.random.choice(len(data["garment_pcd"]), per_pcd, replace=False)
            g_list.append(torch.from_numpy(data["garment_pcd"][ip]).to(torch.float32))
            b_list.append(torch.from_numpy(data["boxmesh_pcd"][ip]).to(torch.float32))
            iq = np.random.choice(len(data["udf"]), per_q, replace=False)
            q_list.append(torch.from_numpy(data["q"][iq]).to(torch.float32))
            u_list.append(torch.from_numpy(data["udf"][iq]).to(torch.float32))
            pair_meta_path = os.path.join(folder, 'pair_meta.yaml')
            if os.path.exists(pair_meta_path):
                # Pre-composed LAYERED pair (scripts/make_pairs.py): one merged
                # sim mesh of top+bottom; design GT from the stored piece designs,
                # layering label from the measured post-sim outcome.
                with open(pair_meta_path) as f:
                    pm = _yaml.safe_load(f)
                t_inner = _load_design_inner(os.path.join(folder, 'top_design.yaml'))
                b_inner = _load_design_inner(os.path.join(folder, 'bottom_design.yaml'))
                design_inners.append(_merge_top_bottom_designs(t_inner, b_inner))
                is_two = True
                top_tucked = bool(pm.get('top_tucked', False))
            else:
                dy = _resolve_design_yaml(folder)
                design_inners.append(_load_design_inner(dy) if dy else None)

        garment_pcd = torch.cat(g_list, dim=0)
        boxmesh_pcd = torch.cat(b_list, dim=0)
        q = torch.cat(q_list, dim=0)
        udf = torch.cat(u_list, dim=0)

        fps_idx = fps_fn(garment_pcd, ratio=1 / 20)
        garment_fps = garment_pcd[fps_idx]
        boxmesh_fps = boxmesh_pcd[fps_idx]

        # combined design GT
        if n == 1:
            inner = design_inners[0]
        else:  # pieces[0] = top (upper), pieces[1] = bottom
            top_i, bot_i = design_inners[0], design_inners[1]
            inner = _merge_top_bottom_designs(top_i, bot_i) if (top_i and bot_i) else (top_i or bot_i)

        if inner is not None:
            gt_parameters = self.design_parameters.create_gt_from_dict(inner)
        else:
            gt_parameters = copy.deepcopy(self.design_parameters)

        if 'is_two_piece' in gt_parameters.panel_data.get('meta', {}):
            gt_parameters.panel_data['meta']['is_two_piece'].set_value(
                torch.tensor([[1.0 if is_two else 0.0]]))
        if 'top_tucked' in gt_parameters.panel_data.get('meta', {}):
            gt_parameters.panel_data['meta']['top_tucked'].set_value(
                torch.tensor([[1.0 if top_tucked else 0.0]]))

        gt_parameters_masked = torch.from_numpy(
            extract_gt_design_params_with_mask(gt_parameters, self.design_parameters))

        return {
            "garment_pcd":          garment_pcd,
            "boxmesh_pcd":          boxmesh_pcd,
            "garment_fps":          garment_fps,
            "boxmesh_fps":          boxmesh_fps,
            "q":                    q,
            "udf":                  udf,
            "garment_name":         "+".join(names),
            "gt_parameters":        gt_parameters,
            "gt_parameters_masked": gt_parameters_masked,
        }

    def _load_sample(self, idx: int) -> dict:
        """Load a single sample from disk without augmentation."""
        if self.new_format:
            return self._load_outfit(self.datapoints_names[idx])
        sample_path = self.datapoints_names[idx]
        basename = os.path.basename(sample_path)
        npz_path = os.path.join(sample_path, f'{basename}_udf_data_0.01.npz')

        data = np.load(npz_path)
        rnd_idx = np.random.choice(len(data["garment_pcd"]), self.pcd_size, replace=False)
        garment_pcd = torch.from_numpy(data["garment_pcd"][rnd_idx]).to(torch.float32)
        boxmesh_pcd = torch.from_numpy(data["boxmesh_pcd"][rnd_idx]).to(torch.float32)

        fps_idx = fps_fn(garment_pcd, ratio=1/20)
        garment_fps = garment_pcd[fps_idx]
        boxmesh_fps = boxmesh_pcd[fps_idx]

        rnd_idx = np.random.choice(len(data["udf"]), self.query_num, replace=False)
        q = torch.from_numpy(data["q"][rnd_idx]).to(torch.float32)
        udf = torch.from_numpy(data["udf"][rnd_idx]).to(torch.float32)

        garment_design_params_path = os.path.join(sample_path, f"{basename}_design_params.yaml")
        gt_parameters = self.design_parameters.create_gt_from_file(garment_design_params_path)

        return {
            "garment_pcd":          garment_pcd,
            "boxmesh_pcd":          boxmesh_pcd,
            "garment_fps":          garment_fps,
            "boxmesh_fps":          boxmesh_fps,
            "q":                    q,
            "udf":                  udf,
            "garment_name":         basename,
            "gt_parameters":        gt_parameters,
            "gt_parameters_masked": torch.from_numpy(
                extract_gt_design_params_with_mask(garment_design_params_path, self.design_parameters)
            ),
        }

    def __getitem__(self, idx):
        try:
            sample = self._load_sample(idx)
        except Exception as e:
            entry = self.datapoints_names[idx]
            desc = entry.get("pieces", entry) if isinstance(entry, dict) else entry
            print(f"Failed to load {desc}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.augmentation is not None:
            second_sample = None
            if self.augmentation.needs_second_sample():
                second_idx = np.random.randint(0, len(self))
                if second_idx != idx:
                    try:
                        second_sample = self._load_sample(second_idx)
                    except Exception:
                        second_sample = None
            sample = self.augmentation.apply(sample, second_sample)

        return sample
