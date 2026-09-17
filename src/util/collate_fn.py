import torch


def _is_design_parameters(obj):
    """Duck-type check for any DesignParameters class (GCD or NT variants)."""
    return hasattr(obj, 'panel_data') and hasattr(obj, 'keys')


def collate_surfd(batch, **kwargs):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        # Check if the value is a tensor
        if torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            out[key] = [item[key] for item in batch]
    return out


def collate_design_parameters(batch):
    collated = batch[0]
    panel_keys = batch[0].keys()
    for panel_name in panel_keys:
        param_keys = batch[0][panel_name].keys()
        for param_name in param_keys:
            collated[panel_name][param_name].join([el[panel_name][param_name] for el in batch[1:]])
    return collated

def collate_pattern_prediction(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        # Check if the value is a tensor
        if torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        elif _is_design_parameters(batch[0][key]):
            all_design_parameters = [item[key] for item in batch]
            out[key] = collate_design_parameters(all_design_parameters)
        else:
            out[key] = [item[key] for item in batch]
    return out