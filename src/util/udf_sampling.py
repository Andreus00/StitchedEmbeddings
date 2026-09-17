import trimesh
import torch
import numpy as np
from typing import Tuple, Union, List
import os

# from util.process_udf import sample_udf_from_mesh


from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import threading
import copy
from typing import Any, Callable, Dict
import tqdm


def load_body_model(body_model_or_path: Union[str, trimesh.Trimesh]):
    """
    Load body model from path.
    """
    if isinstance(body_model_or_path, trimesh.Trimesh):
        return body_model_or_path
    elif isinstance(body_model_or_path, str):
        return trimesh.load(body_model_or_path)
    else:
        raise ValueError(f"Body model not found {body_model_or_path}.")

def _body_model_normalization_params(
    body_model: trimesh.Trimesh,
):
    """
    Load and normalize body model
    """
    translation: np.ndarray = -((body_model.vertices).max(axis=0) / 2 + (body_model.vertices).min(axis=0) / 2)
    scaling: float = 0.009
    return translation, scaling

def _box_normalization_params(mesh: trimesh.Trimesh, absolute_scale):
    b_min, b_max = mesh.bounding_box.bounds[0], mesh.bounding_box.bounds[1]
    translation = -(b_max + b_min) / 2
    scaling = (1 / np.abs(b_max - b_min).max()) * absolute_scale
    return translation, scaling

def normalize_mesh(
    mesh: trimesh.Trimesh, 
    body_model: trimesh.Trimesh="../GarmentCode/garmentcodedata_v2/neutral_body/mean_all.obj", 
    absolute_scale: float = 0.4,
    model_normalization_type: str = "body_model_norm",
    return_parameters: bool = False,
):
    '''
    Normalize a mesh. 
    USe boxmesh normalization by default ("box_norm").

    Use the body model normalization if model_normalization_type = "body_model_norm".

    model_normalization_type: str = ["box_norm", "body_model_norm"]
    '''
    if model_normalization_type == "box_norm":
        translation, scaling = _box_normalization_params(mesh, absolute_scale)
    elif model_normalization_type == "body_model_norm":
        # body_model = load_body_model(body_model)
        # body_model.vertices *= 100
        translation, scaling = np.asarray([0.21533072, -85.99507325, -0.25575086]), 0.009 # _body_model_normalization_params(body_model)
    else:
        raise ValueError(f"Normalization mode {model_normalization_type} does not exist.")
    
    mesh = mesh.apply_translation(translation)
    mesh = mesh.apply_scale(scaling)

    if return_parameters:
        return mesh, translation, scaling
    else:
        return mesh


class Normalizer:
    """
    Garment meshes are 100x larger than the body model.
    The body model is also not centered.

    The Normalizer class normalizes the garment meshes to the body model.
    Then normalizes it again to fit the body model into a box -0.8, 0.8.

    Process:
    during init
    - load the body model
    - normalize the body model
    during normalize
    - normalize the mesh by dividing it by 100
    - centers the mesh
    """
    base_scaling = 1 / 100
    def __init__(self, body_model_path: str="../GarmentCode/garmentcodedata_v2/neutral_body/mean_all.obj", device='cpu'):
        self.body_model = trimesh.load(body_model_path)
        self.translation = -self.body_model.bounding_box.vertices.mean(axis=0)
        self.body_model.apply_translation(self.translation)
        self.device = device
        if device == 'cuda':
            self.translation = torch.tensor(self.translation).to(device).reshape(1, 3).to(torch.float32)
            self.base_scaling = torch.tensor(self.base_scaling).to(device).to(torch.float32)

    def normalize(self, mesh: trimesh.Trimesh):
        base_scaling = self.base_scaling.detach().cpu().numpy() if self.device =='cuda' else self.base_scaling
        translation = self.translation.detach().cpu()[0].numpy() if self.device =='cuda' else self.translation
        # scale by 1/100
        mesh = mesh.apply_scale(base_scaling)
        # center
        mesh = mesh.apply_translation(translation)
        return mesh

    def normalize_pcd(self, pcd: torch.Tensor):
        # scale by 1/100
        base_scaling = self.base_scaling if self.device == 'cuda' else torch.tensor(self.base_scaling, device=pcd.device, dtype=pcd.dtype)
        translation = self.translation if self.device == 'cuda' else torch.tensor(self.translation, device=pcd.device, dtype=pcd.dtype).reshape(1, 3)
        pcd = pcd * base_scaling
        # center
        pcd = pcd + translation
        return pcd

def normalize_mesh(
    mesh: trimesh.Trimesh, 
    body_model: trimesh.Trimesh="../GarmentCode/garmentcodedata_v2/neutral_body/mean_all.obj", 
    model_normalization_type: str = "body_model_norm",
    return_parameters: bool = False,
):
    '''
    Normalize a mesh. 
    USe boxmesh normalization by default ("box_norm").

    Use the body model normalization if model_normalization_type = "body_model_norm".

    model_normalization_type: str = ["box_norm", "body_model_norm"]
    '''
    if model_normalization_type == "box_norm":
        translation, scaling = _box_normalization_params(mesh, absolute_scale)
    elif model_normalization_type == "body_model_norm":
        # body_model = load_body_model(body_model)
        # body_model.vertices *= 100
        translation, scaling = np.asarray([0.21533072, -85.99507325, -0.25575086]), 0.009 # _body_model_normalization_params(body_model)
    else:
        raise ValueError(f"Normalization mode {model_normalization_type} does not exist.")
    
    mesh = mesh.apply_translation(translation)
    mesh = mesh.apply_scale(scaling)

    if return_parameters:
        return mesh, translation, scaling
    else:
        return mesh

# l = [
#     "../../data/GarmentCodeData_v2/GarmentCodeData_v2/garments_5000_0/default_body/data/rand_0A36YXPNV0/rand_0A36YXPNV0_boxmesh.ply",
#     "../../data/GarmentCodeData_v2/GarmentCodeData_v2/garments_5000_0/default_body/data/rand_0A36YXPNV0/rand_0A36YXPNV0_sim.ply",
    
#     "../../data/GarmentCodeData_v2/GarmentCodeData_v2/garments_5000_0/default_body/data/rand_0BC49T2K2O/rand_0BC49T2K2O_boxmesh.ply",
#     "../../data/GarmentCodeData_v2/GarmentCodeData_v2/garments_5000_0/default_body/data/rand_0BC49T2K2O/rand_0BC49T2K2O_sim.ply",
# ]
# for f_idx in range(len(l)):
#     f = l[f_idx]
#     # f = "data/GCD_Mesh/rand_1VTJGQ86O0_sim.ply"
#     m = trimesh.load(f)
#     m, translation, scaling = normalize_mesh(m, model_normalization_type="body_model_norm", return_parameters=True)

#     body_model = load_body_model("../GarmentCode/garmentcodedata_v2/neutral_body/mean_all.obj")
#     body_model.vertices *= 100
#     body_model = body_model.apply_translation(translation)
#     body_model = body_model.apply_scale(scaling)

#     scene = trimesh.Scene()
#     scene.add_geometry(m)
#     scene.add_geometry(body_model)
#     scene.export(f"combined{f_idx}.obj")

def extract_udf_data_from_folder(folder: str, 
    device: str,
    body_model=None, 
    absolute_scale: float = 0.4, 
    model_normalization_type: str = "box_norm",
 ) -> None:
    """
    Load the _sim.ply file and the _boxmesh.ply file.
    Then run the sample_udf_from_mesh on both and save the results in a .npz file.
    """
    garment_name = os.path.basename(folder)

    # load sim model
    sim_path = os.path.join(folder, f"{garment_name}_sim.ply")
    sim_mesh = trimesh.load(sim_path)
    sim_mesh = normalize_mesh(sim_mesh, body_model=body_model, absolute_scale=absolute_scale, model_normalization_type=model_normalization_type)

    # load boxmesh model
    boxmesh_path = os.path.join(folder, f"{garment_name}_boxmesh.ply")
    boxmesh_mesh = trimesh.load(boxmesh_path)
    boxmesh_mesh = normalize_mesh(boxmesh_mesh, body_model=body_model, absolute_scale=absolute_scale, model_normalization_type=model_normalization_type)

    sim_results = sample_udf_from_mesh(sim_mesh, 250_000, device=device)
    boxmesh_results = sample_udf_from_mesh(boxmesh_mesh, 250_000, device=device)
    
    np.savez(
        os.path.join(folder, "surfd_udf_samples.npz"),
        sim_surface=sim_results[0],
        sim_surface_grads=sim_results[1],
        sim_boundary=sim_results[2],
        sim_boundary_grads=sim_results[3],
        sim_importance_points=sim_results[4],
        sim_importance_grads=sim_results[5],
        sim_coords_near=sim_results[6],
        sim_udf_near=sim_results[7],
        sim_gradients_near=sim_results[8],
        sim_coords_rand=sim_results[9],
        sim_udf_rand=sim_results[10],
        sim_gradients_rand=sim_results[11],
        boxmesh_surface=boxmesh_results[0],
        boxmesh_surface_grads=boxmesh_results[1],
        boxmesh_boundary=boxmesh_results[2],
        boxmesh_boundary_grads=boxmesh_results[3],
        boxmesh_importance_points=boxmesh_results[4],
        boxmesh_importance_grads=boxmesh_results[5],
        boxmesh_coords_near=boxmesh_results[6],
        boxmesh_udf_near=boxmesh_results[7],
        boxmesh_gradients_near=boxmesh_results[8],
        boxmesh_coords_rand=boxmesh_results[9],
        boxmesh_udf_rand=boxmesh_results[10],
        boxmesh_gradients_rand=boxmesh_results[11],
    )
    # Clean up cache if needed
    torch.cuda.empty_cache()


class DistributedSurfDSingleton:

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def model_to_gpu(cls, vae, gpu_id):
        """
        If there is a model for the gpu_id, return it. Otherwise clone the model on the required GPU.
        """
        with cls._lock:
            if gpu_id not in cls._instances:
                model_copy = copy.deepcopy(vae)
                cls._instances[gpu_id] = model_copy.to(gpu_id)
        return cls._instances[gpu_id]
            

def _extract_latent_from_folder_list(garment_folders: str, gpu_id: int, vae):
    """Process a single garment folder on the assigned GPU."""
    torch.cuda.set_device(gpu_id)
    gpu_id = f"cuda:{gpu_id}"

    # Send model to GPU.
    vae = DistributedSurfDSingleton.model_to_gpu(vae=vae, gpu_id=gpu_id)

    garment_names = [os.path.basename(garment_folder) for garment_folder in garment_folders]
    end_name = "latent_shape_surfd.npz"
    latent_shape_files = [os.path.join(garment_folder, end_name) for i, garment_folder in enumerate(garment_folders)]

    garment_files = [os.path.join(garment_folder, f'{garment_names[i]}_sim.ply') for i, garment_folder in enumerate(garment_folders)]

    pcds = []
    for file in garment_files:
        m = trimesh.load(file)
        m = normalize_mesh(mesh=m)
        pcds.append(torch.asarray(trimesh.sample.sample_surface(m, 10_000)[0]).reshape(1, 10_000, 3).to("cuda").float())
        
    pcds = torch.cat(pcds, dim=0)

    with torch.no_grad():
        latent_shapes = vae.encode(pcds)
    
    for idx in range(len(latent_shapes)):
        np.savez(latent_shape_files[idx], latent_shape=latent_shapes[idx].detach().cpu().numpy())
    del latent_shapes

def _extract_udf_from_folder_list(
    folder_list: List[str],
    gpu_id: int,
    device: str,
    pbar: bool = False
) -> None:
    if pbar:
        folder_list = tqdm.tqdm(folder_list, total=len(folder_list), desc="Extracting UDF")
    if device == "cuda":
        torch.cuda.set_device(gpu_id)
    for folder in folder_list:
        extract_udf_data_from_folder(folder, device=device)


def _parallel_gpu_run(input_list: List[Any], _function: Callable, batch_size: int, num_workers: int, **kwds):
    """
    Map a function on multiple gpus.
    """
    # create batches
    batch_size = min(batch_size, len(input_list))
    batches = [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    jobs = [(garment_batch, i % num_workers) for i, garment_batch in enumerate(batches)]

    # Wrap the function to bind parameters
    wrapped_func = partial(_function, **kwds)

    # Run with threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(wrapped_func, batch, gpu_id) for batch, gpu_id in jobs]
        for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Extracting using {_function}"):
            pass

def extract_latents_from_shape(garment_folders: List[str], vae, num_workers: int, batch_size):
    """
    Run latent code extraction across multiple GPUs.
    """

    end_name = "latent_shape_surfd.npz"
    latent_shape_files = [os.path.join(garment_folder, end_name) for i, garment_folder in enumerate(garment_folders)]

    # pop already processed files
    for idx, latent_shape_file in tqdm.tqdm(reversed(list(enumerate(latent_shape_files))), desc="removing already processed files", total=len(latent_shape_files)):
        if os.path.exists(latent_shape_file):
            garment_folders.pop(idx)
            latent_shape_files.pop(idx)
    print(f"Garments withtout latent_shape file: {len(garment_folders)}")

    if len(garment_folders) > 0:
        _parallel_gpu_run(input_list=garment_folders, 
                        _function=_extract_latent_from_folder_list, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        vae=vae)
        

def extract_udf(garment_folders: List[str], num_workers: int, batch_size: int, device: str, start_from: float=0.):
    """
    Run udf extraction across multiple GPUs.
    """

    start = int(start_from * len(garment_folders))
    print(f"Starting from garment number {start} / {len(garment_folders)}")
    garment_folders = garment_folders[start:]
    end_name = "surfd_udf_samples.npz"
    udf_files = [os.path.join(garment_folder, end_name) for i, garment_folder in enumerate(garment_folders)]

    # pop already processed files
    for idx, udf_file in tqdm.tqdm(reversed(list(enumerate(udf_files))), desc="removing already processed files", total=len(udf_files)):
        if os.path.exists(udf_file):
            try:
                with np.load(udf_file):
                    pass
            except Exception:   # found corrupted file
                continue
            garment_folders.pop(idx)
            udf_files.pop(idx)
    print(f"Garments withtout udf file: {len(garment_folders)}")

    if len(garment_folders) > 0:
        print(f"Running _extract_udf_from_folder_list on {num_workers} with batch_size {batch_size}")
        # _extract_udf_from_folder_list(garment_folders, 0, device=device)
        _parallel_gpu_run(input_list=garment_folders, 
                        _function=_extract_udf_from_folder_list, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        device=device)
        

def sample_from_udf_data(udf_data: Dict[str, np.ndarray], sfc_pts: int, imp_pts: int, bnd_pts: int, query_pts: int, grads=False):
    """
    Sample points from the udf_data dictionary according to the provided numbers.
    For coords_rand and coords_near, treat them as a single concatenated array for sampling.
    Returns a dictionary with the sampled points and their corresponding gradients/udf values.
    """
    out = {}

    for pre in ["sim", "boxmesh"]:
        # Surface points
        surface = udf_data[f"{pre}_surface"]
        surface_grads = udf_data[f"{pre}_surface_grads"]
        idx = np.random.choice(surface.shape[0], sfc_pts, replace=surface.shape[0] < sfc_pts)
        surface = surface[idx]
        surface_grads = surface_grads[idx]

        if imp_pts > 0:
            # # Importance points
            imp = udf_data[f"{pre}_importance_points"]
            imp_grads = udf_data[f"{pre}_importance_grads"]
            idx = np.random.choice(imp.shape[0], imp_pts, replace=imp.shape[0] < imp_pts)
            surface = np.concatenate([surface, imp[idx]], axis=0)
            surface_grads = np.concatenate([surface_grads, imp_grads[idx]], axis=0)
        
        
        if bnd_pts > 0:
            # # Importance points
            bnd = udf_data[f"{pre}_boundary"]
            bnd_grads = udf_data[f"{pre}_boundary_grads"]
            idx = np.random.choice(bnd.shape[0], bnd_pts, replace=bnd.shape[0] < bnd_pts)
            surface = np.concatenate([surface, bnd[idx]], axis=0)
            surface_grads = np.concatenate([surface_grads, bnd_grads[idx]], axis=0)

        out[f"{pre}_surface"] = torch.asarray(surface).float()
        if grads:
            out[f"{pre}_surface_grads"] = torch.asarray(surface_grads).float()

        # # Boundary points
        # bnd = udf_data[f"{pre}_boundary"]
        # bnd_grads = udf_data[f"{pre}_boundary_grads"]
        # idx = np.random.choice(bnd.shape[0], bnd_pts, replace=bnd.shape[0] < bnd_pts)
        # out[f"{pre}_boundary"] = torch.asarray(bnd[idx])
        # out[f"{pre}_boundary_grads"] = torch.asarray(bnd_grads[idx])

        # Query points (from coords_near and coords_rand concatenated)
        coords_near = udf_data[f"{pre}_coords_near"]
        udf_near = udf_data[f"{pre}_udf_near"]
        grads_near = udf_data[f"{pre}_gradients_near"]
        idx = np.random.choice(coords_near.shape[0], query_pts, replace=coords_near.shape[0] < query_pts)
        coords_near = coords_near[idx]
        udf_near = udf_near[idx]
        grads_near = grads_near[idx]

        coords_rand = udf_data[f"{pre}_coords_rand"]
        udf_rand = udf_data[f"{pre}_udf_rand"]
        grads_rand = udf_data[f"{pre}_gradients_rand"]
        idx = np.random.choice(coords_rand.shape[0], query_pts, replace=coords_rand.shape[0] < query_pts)
        coords_rand = coords_rand[idx]
        udf_rand = udf_rand[idx]
        grads_rand = grads_rand[idx]

        coords_all = np.concatenate([coords_near, coords_rand], axis=0)
        udf_all = np.concatenate([udf_near, udf_rand], axis=0)
        grads_all = np.concatenate([grads_near, grads_rand], axis=0)


        out[f"{pre}_query_coords"] = torch.asarray(coords_all)
        out[f"{pre}_query_udf"] = torch.asarray(udf_all)
        if grads:
            out[f"{pre}_query_grads"] = torch.asarray(grads_all)

    return out



def remove_stitching_triangles(boxmesh_trimesh, garment_trimesh, threshold = 2.):
    """
    Remove triangles that belong to the pattern stitches.

    Uses an heuristics based on removing triangles with long edges.
    """
    verts_boxmesh = np.asarray(boxmesh_trimesh.vertices)   # Nx3
    faces_boxmesh = np.asarray(boxmesh_trimesh.faces)  # Nx3
    verts_garment = np.asarray(garment_trimesh.vertices)   # Nx3

    faces_with_edges = verts_boxmesh[faces_boxmesh]    # Nx3xx3

    def lp_distance(a1: np.ndarray, a2: np.ndarray, p: int = 2):
        # expected a1.shape = a2.shape = N x 3
        assert a1.shape == a2.shape, f"Arrays have different shape. a1({a1.shape}) != a2({a2.shape})"
        return np.sqrt(np.sum(np.power(a1 - a2, p), axis=-1, keepdims=True))  # returns Nx1, where each entry is the n-th Lp distance

    faces_edges_length = np.concatenate(
        [lp_distance(faces_with_edges[:, 0, :], faces_with_edges[:, 1, :]), 
        lp_distance(faces_with_edges[:, 1, :], faces_with_edges[:, 2, :]), 
        lp_distance(faces_with_edges[:, 2, :], faces_with_edges[:, 0, :]),],
        axis=-1
    )

    mask = ~np.any(faces_edges_length > threshold, axis=-1)

    filtered_faces = faces_boxmesh[mask]

    boxmesh_filtered = trimesh.Trimesh(verts_boxmesh, faces=filtered_faces)
    garment_filtered = trimesh.Trimesh(verts_garment, faces=filtered_faces)

    return boxmesh_filtered, garment_filtered