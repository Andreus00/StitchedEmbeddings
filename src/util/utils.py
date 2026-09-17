import numpy as np
import torch
import random

def rotTransltoMatrix(rot, transl):
    """
    Convert rotation and translation to a 4x4 matrix
    """
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = transl
    return mat


def collate_fn(batch, tokenizer):
    latent_list = []
    encoded_pattern_list = []
    pattern_param_list = []
    pattern_endpoints_list = []
    pattern_transf_list = []
    names = []
    sfc_list = []

    for pattern_dict, latent in batch:
        names.append(pattern_dict["name"])
        sfc_list.append(pattern_dict["sfc"])
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            pattern_param_list.append([{}])
        pattern_endpoint = pattern_dict.get("endpoints", None)
        if pattern_endpoint is not None:
            pattern_endpoint = torch.cat(pattern_endpoint)
        else:
            pattern_endpoint = torch.zeros(0, 2)
        pattern_endpoints_list.append(pattern_endpoint)
        pattern_transf = pattern_dict.get("transformations", None)
        if pattern_transf is not None:
            pattern_transf = torch.cat(pattern_transf)
        else:
            pattern_transf = torch.zeros(0, 7)
        pattern_transf_list.append(pattern_transf)
        
        latent_list.append(latent)
    

    pattern_ids = [torch.as_tensor(tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids[0], dtype=torch.int) if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        pattern_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    endpoint_max_len = max([len(endpoints) for endpoints in pattern_endpoints_list])
    pattern_endpoints = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, 2, dtype=torch.float32)
    pattern_endpoint_masks = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, dtype=torch.bool)
    for i, endpoints in enumerate(pattern_endpoints_list):
        if len(endpoints) == 0:
            continue
        pattern_endpoints[i, :len(endpoints)] = endpoints
        pattern_endpoint_masks[i, :len(endpoints)] = True
    pattern_transf_max_len = max([len(transf) for transf in pattern_transf_list])
    pattern_transfs = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, pattern_transf_list[0].shape[-1], dtype=torch.float32)
    pattern_transf_masks = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, dtype=torch.bool)
    for i, transf in enumerate(pattern_transf_list):
        if len(transf) == 0:
            continue
        pattern_transfs[i, :len(transf)] = transf
        pattern_transf_masks[i, :len(transf)] = True
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    input_len = input_ids.shape[1]
    latent_shapes = torch.stack(latent_list, dim=0).squeeze(1)

    out_dict = {
        "input_len": input_len,
        "param_target_endpoints": pattern_endpoints,
        "param_target_endpoints_mask": pattern_endpoint_masks,
        "param_target_transformations": pattern_transfs,
        "param_target_transformations_mask": pattern_transf_masks,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "latent_shapes": latent_shapes,
        "name": names,
        "sfc": sfc_list,
        }
    return out_dict



def collate_with_points_fn(batch, tokenizer):
    udfs_list = []
    points_list = []
    # surface_list = []
    encoded_pattern_list = []
    pattern_param_list = []
    pattern_endpoints_list = []
    pattern_transf_list = []
    names = []
    sfc_list = []

    for pattern_dict, udfs in batch:
        names.append(pattern_dict["name"])
        sfc_list.append(pattern_dict["sfc"])
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            pattern_param_list.append([{}])
        pattern_endpoint = pattern_dict.get("endpoints", None)
        if pattern_endpoint is not None:
            pattern_endpoint = torch.cat(pattern_endpoint)
        else:
            pattern_endpoint = torch.zeros(0, 2)
        pattern_endpoints_list.append(pattern_endpoint)
        pattern_transf = pattern_dict.get("transformations", None)
        if pattern_transf is not None:
            pattern_transf = torch.cat(pattern_transf)
        else:
            pattern_transf = torch.zeros(0, 7)
        pattern_transf_list.append(pattern_transf)
        
        pattern_points = pattern_dict.get("query_points", None)
        if pattern_points is not None:
            points_list.append(pattern_points)
        else:
            raise ValueError("Query points are required in the pattern dictionary.")

        # surface_points = pattern_dict.get("surface_points", None)
        # if surface_points is not None:
        #     surface_list.append(surface_points)

        udfs_list.append(udfs)


    pattern_ids = [torch.as_tensor(tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids[0], dtype=torch.int) if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        pattern_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    endpoint_max_len = max([len(endpoints) for endpoints in pattern_endpoints_list])
    pattern_endpoints = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, 2, dtype=torch.float32)
    pattern_endpoint_masks = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, dtype=torch.bool)
    for i, endpoints in enumerate(pattern_endpoints_list):
        if len(endpoints) == 0:
            continue
        pattern_endpoints[i, :len(endpoints)] = endpoints
        pattern_endpoint_masks[i, :len(endpoints)] = True
    pattern_transf_max_len = max([len(transf) for transf in pattern_transf_list])
    pattern_transfs = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, pattern_transf_list[0].shape[-1], dtype=torch.float32)
    pattern_transf_masks = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, dtype=torch.bool)
    for i, transf in enumerate(pattern_transf_list):
        if len(transf) == 0:
            continue
        pattern_transfs[i, :len(transf)] = transf
        pattern_transf_masks[i, :len(transf)] = True
    attention_masks = input_ids.ne(tokenizer.pad_token_id)


    input_len = input_ids.shape[1]

    points_list = torch.stack(points_list, dim=0).squeeze(1)
    udfs_list = torch.stack(udfs_list, dim=0).squeeze(1)
    # surface_list = torch.stack(surface_list, dim=0).squeeze(1)

    out_dict = {
        "input_len": input_len,
        "param_target_endpoints": pattern_endpoints,
        "param_target_endpoints_mask": pattern_endpoint_masks,
        "param_target_transformations": pattern_transfs,
        "param_target_transformations_mask": pattern_transf_masks,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "query_points": points_list,
        "udfs": udfs_list,
        "name": names,
        "sfc": sfc_list,
        }
    return out_dict



def collate_both_fn(batch, tokenizer):
    udfs_list = []
    points_list = []
    latent_list = []
    encoded_pattern_list = []
    pattern_param_list = []
    pattern_endpoints_list = []
    pattern_transf_list = []
    names = []
    sfc_list = []

    for pattern_dict, (latent, udfs) in batch:
        names.append(pattern_dict["name"])
        sfc_list.append(pattern_dict["sfc"])
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            pattern_param_list.append([{}])
        pattern_endpoint = pattern_dict.get("endpoints", None)
        if pattern_endpoint is not None:
            pattern_endpoint = torch.cat(pattern_endpoint)
        else:
            pattern_endpoint = torch.zeros(0, 2)
        pattern_endpoints_list.append(pattern_endpoint)
        pattern_transf = pattern_dict.get("transformations", None)
        if pattern_transf is not None:
            pattern_transf = torch.cat(pattern_transf)
        else:
            pattern_transf = torch.zeros(0, 7)
        pattern_transf_list.append(pattern_transf)
        
        pattern_points = pattern_dict.get("query_points", None)
        if pattern_points is not None:
            points_list.append(pattern_points)
        else:
            raise ValueError("Query points are required in the pattern dictionary.")

        latent_list.append(latent)
        udfs_list.append(udfs)
        


    pattern_ids = [torch.as_tensor(tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids[0], dtype=torch.int) if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        pattern_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    endpoint_max_len = max([len(endpoints) for endpoints in pattern_endpoints_list])
    pattern_endpoints = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, 2, dtype=torch.float32)
    pattern_endpoint_masks = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, dtype=torch.bool)
    for i, endpoints in enumerate(pattern_endpoints_list):
        if len(endpoints) == 0:
            continue
        pattern_endpoints[i, :len(endpoints)] = endpoints
        pattern_endpoint_masks[i, :len(endpoints)] = True
    pattern_transf_max_len = max([len(transf) for transf in pattern_transf_list])
    pattern_transfs = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, pattern_transf_list[0].shape[-1], dtype=torch.float32)
    pattern_transf_masks = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, dtype=torch.bool)
    for i, transf in enumerate(pattern_transf_list):
        if len(transf) == 0:
            continue
        pattern_transfs[i, :len(transf)] = transf
        pattern_transf_masks[i, :len(transf)] = True
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    input_len = input_ids.shape[1]

    points_list = torch.stack(points_list, dim=0).squeeze(1)
    udfs_list = torch.stack(udfs_list, dim=0).squeeze(1)
    latent_list = torch.stack(latent_list, dim=0).squeeze(1)
    # surface_list = torch.stack(surface_list, dim=0).squeeze(1)

    out_dict = {
        "input_len": input_len,
        "param_target_endpoints": pattern_endpoints,
        "param_target_endpoints_mask": pattern_endpoint_masks,
        "param_target_transformations": pattern_transfs,
        "param_target_transformations_mask": pattern_transf_masks,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "query_points": points_list,
        "udfs": udfs_list,
        "latent_shapes": latent_list,
        "name": names,
        "sfc": sfc_list,
        }
    return out_dict



def collate_fn_wrapper(batch, tokenizer, predict_latents, predict_udf):
    return collate_both_fn(batch, tokenizer)
    # if predict_latents and predict_udf:
    #     return collate_both_fn(batch, tokenizer)
    # elif predict_latents:
    #     return collate_fn(batch, tokenizer)
    # else:
    #     return collate_with_points_fn(batch, tokenizer)


def shuffle_list(input_list, seed):
    random.Random(seed).shuffle(input_list)
    return input_list




def collate_v2(batch, tokenizer, predict_latents, predict_udf):
    assert predict_latents or predict_udf, "Error in collate: predict latents and predict udf are both false"

    udfs_list = []
    points_list = []
    latent_list = []
    encoded_pattern_list = []
    pattern_param_list = []
    params2tokens_list = []
    pattern_transf_list = []
    names = []
    sfc_list = []

    for pattern_dict, labels in batch:
        names.append(pattern_dict["name"])
        sfc_list.append(pattern_dict["sfc"])
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            raise Error("Missing pattern parameters")

        params2tokens = pattern_dict.get("params2tokens", None)
        if params2tokens is not None:
            params2tokens_list.append(params2tokens)
        else:
            raise Error("Missing params2tokens")
        
        
        if predict_udf:
            pattern_points = pattern_dict.get("query_points", None)
            if pattern_points is not None:
                points_list.append(pattern_points)
            else:
                raise ValueError("Query points are required in the pattern dictionary.")
            udfs_list.append(labels[1])

        latent_list.append(labels[0])
                
    pattern_ids = [torch.as_tensor(tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids, dtype=torch.int) if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        pattern_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    input_len = input_ids.shape[1]

    def pad_and_mask(tensor_list, pad_value=0.0):
        # 1. Find max length
        max_len = max(len(t) for t in tensor_list)
        batch_size = len(tensor_list)

        # 2. Init padded tensor and mask
        padded = torch.full((batch_size, max_len, tensor_list[0].size(-1)),
                            pad_value, dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # 3. Fill
        for i, t in enumerate(tensor_list):
            length = t.size(0)
            padded[i, :length] = t
            mask[i, :length] = True   # True = real token

        return padded, mask

    # Pad pattern_param_list to the same length
    pattern_param_list, pattern_param_mask = pad_and_mask(pattern_param_list)
    pattern_param_list /= 100

    # max_param_len = max(len(params) for params in params2tokens_list)
    params2tokens_list, params2tokens_mask = params2tokens_list, None

    

    out_dict = {
        "input_len": input_len,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "pattern_param": pattern_param_list,
        "params2tokens": params2tokens_list,
        "pattern_param_mask": pattern_param_mask,
        "name": names,
        "sfc": sfc_list,
    }
    if predict_udf:
        out_dict["query_points"] = torch.stack(points_list, dim=0).squeeze(1)
        out_dict["udfs"] = torch.stack(udfs_list, dim=0).squeeze(1)
    out_dict["latent_shapes"] = torch.stack(latent_list, dim=0).squeeze(1)


    return out_dict



def collate_hy3d(batch, tokenizer, predict_latents, predict_udf):
    assert predict_latents or predict_udf, "Error in collate: predict latents and predict udf are both false"

    points_list = []
    encoded_pattern_list = []
    pattern_param_list = []
    params2tokens_list = []
    pattern_transf_list = []
    names = []
    sim_surface_list = []
    boxmesh_surface_list = []
    query_points_sim_list = []
    query_points_boxmesh_list = []

    for pattern_dict in batch:
        names.append(pattern_dict["name"])
        sim_surface_list.append(pattern_dict["sim_surface"])
        boxmesh_surface_list.append(pattern_dict["boxmesh_surface"])
        query_points_sim_list.append(pattern_dict["query_points_sim"])
        query_points_boxmesh_list.append(pattern_dict["query_points_boxmesh"])
        
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            raise Error("Missing pattern parameters")

        params2tokens = pattern_dict.get("params2tokens", None)
        if params2tokens is not None:
            params2tokens_list.append(params2tokens)
        else:
            raise Error("Missing params2tokens")

                
    pattern_ids = [torch.as_tensor(tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids, dtype=torch.int) if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        pattern_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    input_len = input_ids.shape[1]

    def pad_and_mask(tensor_list, pad_value=0.0):
        # 1. Find max length
        max_len = max(len(t) for t in tensor_list)
        batch_size = len(tensor_list)

        # 2. Init padded tensor and mask
        padded = torch.full((batch_size, max_len, tensor_list[0].size(-1)),
                            pad_value, dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # 3. Fill
        for i, t in enumerate(tensor_list):
            length = t.size(0)
            padded[i, :length] = t
            mask[i, :length] = True   # True = real token

        return padded, mask

    # Pad pattern_param_list to the same length
    pattern_param_list, pattern_param_mask = pad_and_mask(pattern_param_list)
    pattern_param_list /= 100

    # sfc_list = torch.stack(sfc_list, dim=0).squeeze(1)

    out_dict = {
        "input_len": input_len,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "pattern_param": pattern_param_list,
        "params2tokens": params2tokens_list,
        "pattern_param_mask": pattern_param_mask,
        "name": names,
        "sim_surface": sim_surface_list,
        "boxmesh_surface": boxmesh_surface_list,
        "query_points_sim": query_points_sim_list,
        "query_points_boxmesh": query_points_boxmesh_list,
    }


    return out_dict


    # max_param_len = max(len(params) for params in params2tokens_list)
    params2tokens_list, params2tokens_mask = params2tokens_list, None
    sim_surface_list = torch.stack(sim_surface_list, dim=0).squeeze(1)
    boxmesh_surface_list = torch.stack(boxmesh_surface_list, dim=0).squeeze(1)
    query_points_sim_list = torch.stack(query_points_sim_list, dim=0).squeeze(1)
    query_points_boxmesh_list = torch.stack(query_points_boxmesh_list, dim=0).squeeze(1)
    # sfc_list = torch.stack(sfc_list, dim=0).squeeze(1)

    out_dict = {
        "input_len": input_len,
        "attention_masks": attention_masks,
        "input_ids": input_ids,
        "pattern_param": pattern_param_list,
        "params2tokens": params2tokens_list,
        "pattern_param_mask": pattern_param_mask,
        "name": names,
        "sim_surface": sim_surface_list,
        "boxmesh_surface": boxmesh_surface_list,
        "query_points_sim": query_points_sim_list,
        "query_points_boxmesh": query_points_boxmesh_list,
    }


    return out_dict




def collate_surfd(batch, **kwargs):
    names = []
    sim_surface_list = []
    boxmesh_surface_list = []
    gt_latent_list = []

    for pattern_dict in batch:
        names.append(pattern_dict["name"])
        sim_surface_list.append(pattern_dict["sim_surface"])
        boxmesh_surface_list.append(pattern_dict["boxmesh_surface"])
        gt_latent_list.append(pattern_dict["gt_latent"])

    # max_param_len = max(len(params) for params in params2tokens_list)
    sim_surface_list = torch.stack(sim_surface_list, dim=0).squeeze(1)
    boxmesh_surface_list = torch.stack(boxmesh_surface_list, dim=0).squeeze(1)
    gt_latent_list = torch.stack(gt_latent_list, dim=0).squeeze(1)

    out_dict = {
        "name": names,
        "sim_surface": sim_surface_list,
        "boxmesh_surface": boxmesh_surface_list,
        "gt_latents": gt_latent_list,
    }

    return out_dict
