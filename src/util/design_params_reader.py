import yaml
import torch
import copy
import os
from functools import reduce  # forward compatibility for Python 3
import operator
from pathlib import Path
import tqdm
from functools import reduce
import operator
import json

import sys
# sys.path.append('/home/andrea/Documents/PhD/Projects/LLM-SP-ENC/galvani/mesh_patterns/src/external/NeuralTailor')
# from dataset_yaml_io import load_spec_to_yaml_format, save_yaml_dict_as_spec


# from src.external.GarmentCodeAssets.assets.differentiable_garment_programs.meta_garment import DifferentiableMetaGarment
# from src.external.GarmentCodeAssets.assets.bodies.body_params import BodyParameters

from src.external.GarmentCode.pattern_sampler import _save_sample, has_pants
from src.external.GarmentCode.assets.garment_programs.meta_garment import MetaGarment
from src.external.GarmentCode.assets.bodies.body_params import BodyParameters

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

class DefauldRegressionHead(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DefauldRegressionHead, self).__init__()
        # self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = self.fc(x)
        return x

class BinaryRegressionHead(torch.nn.Module):
    # used for Bool parameters
    def __init__(self, input_dim, output_dim):
        super(BinaryRegressionHead, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_dim // 2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class RangeRegressionHead(torch.nn.Module):
    # used for Float and Integer parameters
    def __init__(self, input_dim, output_dim, min_val=0, max_val=1):
        super(RangeRegressionHead, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_dim // 2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.min_val = min_val
        self.max_val = max_val

    def value_to_range(self, x):
        return x * (self.max_val - self.min_val) + self.min_val
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # x = self.value_to_range(x)
        return x
    

class MulticlassRegressionHead(torch.nn.Module):
    # used for Select parameters
    def __init__(self, input_dim, output_dim):
        super(MulticlassRegressionHead, self).__init__()
        self.num_classes = output_dim
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_dim // 2, self.num_classes)
        # self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x
    

class DefaultParameter:

    def __init__(self, name, param_info):
        self.name = name
        self.value = self._setup_value(param_info)
        self.setup_loss_function()
        self.type = None

    def get_name(self):
        return self.name

    def get_options(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_output_size(self):
        return 1
    
    def get_options(self):
        return {}
    
    def get_value(self):
        return self.value
    
    def _setup_value(self, param_info):
        return torch.asarray([param_info['v']]).reshape(1, -1).float()
    
    def setup_loss_function(self):
        self.loss_function = torch.nn.MSELoss()
    
    def loss(self, other_param):
        return self.loss_function(other_param, self.get_value())
    
    def value_of(self, other_parameter):
        # for non-categorical parameters, the value is directly the input
        return other_parameter.get_value()
    
    def set_value(self, new_value):
        self.value = new_value

    def join(self, other_parameters):
        all_values = [self.get_value()]
        for other_param in other_parameters:
            all_values.append(other_param.get_value())
        self.value = torch.cat(all_values, dim=0)

    def to(self, device):
        self.value = self.value.to(device)
        return self
    
    def prediction_to_value(self, prediction, differentiable=False):
        return (prediction.reshape(-1,)).tolist()
        
    def __repr__(self):
        return f"{self.__class__} - value: {self.value}"
    
    def serialize(self):
        return {'v': self.prediction_to_value(self.value)[0], 'type': self.type}

class FloatParameter(DefaultParameter):
    
    def __init__(self, name, param_info):
        self.min_value = param_info['range'][0]
        self.max_value = param_info['range'][1]
        super().__init__(name, param_info)
        self.type = "float"
    
    def _setup_value(self, param_info):
        return (torch.asarray([param_info['v']], dtype=torch.float32).reshape(1, -1).float() - self.min_value) / (self.max_value - self.min_value)
    
    def sigmoid_to_range(self, sigmoid_output):
        return sigmoid_output * (self.max_value - self.min_value) + self.min_value

    def prediction_to_value(self, prediction, differentiable=False):
        return self.sigmoid_to_range(prediction).reshape(-1,) if differentiable else self.sigmoid_to_range(prediction).reshape(-1,).tolist()
    
    def get_options(self):
        return {
            'min_val': self.min_value,
            'max_val': self.max_value
        }

    def serialize(self):
        d = super().serialize()
        d['range'] = [self.min_value,  self.max_value]
        return d
    

class IntParameter(FloatParameter):
    
    def __init__(self, name, param_info):
        super().__init__(name, param_info)
        self.type = "int"

    def get_options(self):
        return {
            'min_val': self.min_value,
            'max_val': self.max_value
        }
    
    def sigmoid_to_range(self, sigmoid_output):
        return torch.round(super().sigmoid_to_range(sigmoid_output)).to(torch.int8)
    
    
class BoolParameter(DefaultParameter):
    
    def __init__(self, name, param_info):
        super().__init__(name, param_info)
        self.type = "bool"

    def get_options(self):
        return super().get_options()
    
    def setup_loss_function(self):
        self.loss_function = torch.nn.BCELoss()
    
    def prediction_to_value(self, prediction, differentiable=False):
        return prediction.reshape(-1,) > 0.5 if differentiable else (prediction.reshape(-1,) > 0.5).tolist()


class SelectParameter(DefaultParameter):

    def __init__(self, name, param_info):
        self.setup_options(param_info)
        super().__init__(name, param_info)
        self.type = "select"

    def _setup_value(self, param_info):
        self._value_str = param_info['v']
        return self.one_hot_encode(self._value_str).reshape(1, -1)

    def setup_options(self, param_info):
        self.options = param_info['range']

    def one_hot_encode(self, value):
        one_hot = torch.zeros(len(self.options))
        if value in self.options:
            index = self.options.index(value)
            one_hot[index] = 1
        if one_hot.sum() != 1:
            print(f"Value '{value}'error in options {self.options} for parameter '{self.name}'. Sum is {one_hot.sum()}")
        return one_hot
    
    def setup_loss_function(self):
        self.loss_function = torch.nn.CrossEntropyLoss()

    def loss(self, other_param):
        return self.loss_function(other_param, self.value.argmax(dim=1))

    def get_options(self):
        return {
            # 'options': self.options
        }
    
    def get_output_size(self):
        return len(self.options)
    
    def value_of(self, other_parameter: DefaultParameter):
        # for categorical parameters, decode the one-hot encoding
        return self.one_hot_encode(other_parameter._value_str).reshape(1, -1)
    
    def set_value(self, new_value):
        self.value = new_value

    def prediction_to_value(self, prediction, differentiable=False):
        index = torch.argmax(prediction, dim=-1).reshape(-1,)
        # if differentiable:
        #     return index
        ret = []
        for i in range(len(index)):
            ret.append(self.options[index[i]])
        return ret

    
    def join(self, other_parameters):
        super().join(other_parameters)
        all_value_str = [self._value_str]
        for other_param in other_parameters:
            all_value_str.append(other_param._value_str)
        self._value_str = all_value_str

    def __repr__(self):
        return f"{self.__class__} - value: {self._value_str}"


    def serialize(self):
        d = super().serialize()
        d['range'] = self.options
        return d
    
class SelectNullParameter(SelectParameter):

    def __init__(self, name, param_info):
        super().__init__(name, param_info)
        self.type = "select_null"

    def setup_options(self, param_info):
        super().setup_options(param_info)
        # force None to be present as last option
        if None in self.options:
            self.options.remove(None)
        self.options.append(None)

    def _setup_value(self, param_info):
        self._value_str = param_info['v']
        return self.one_hot_encode(self._value_str).reshape(1, -1)
    

PARAMETER_TYPE_MAP = {
    'float': FloatParameter,
    'int': IntParameter,
    'bool': BoolParameter,
    'select': SelectParameter,
    'select_null': SelectNullParameter
}


PARAMETER_TO_HEAD_MAP = {
    FloatParameter: RangeRegressionHead,
    IntParameter: RangeRegressionHead,
    BoolParameter: BinaryRegressionHead,
    SelectParameter: MulticlassRegressionHead,
    SelectNullParameter: MulticlassRegressionHead
}

class PanelData:

    def __init__(self, name, panel_info):
        self.name = name
        self.panel_info = panel_info
        self.classification_values = None
        self.setup_classification_values()

    def setup_classification_values(self):
        self.classification_values = {}
        for property, values in self.panel_info.items():
            if 'type' not in values.keys() or property == 'cuff':
                self.classification_values[property] = PanelData(property, values)
            else:
                # leaf parameter
                param_type = values['type']
                param_class = PARAMETER_TYPE_MAP.get(param_type, None)
                if param_class is not None:
                    param_instance = param_class(property, values)
                    self.classification_values[property] = param_instance
                # else:
                #     print(f"Warning: Unknown parameter type '{param_type}' for property '{property}' in panel '{self.name}'")
    
    def flatten(self):
        flat_params = {}
        for param_name, param in self.classification_values.items():
            if isinstance(param, PanelData):
                # nested panel
                nested_flat_params = param.flatten()
                flat_params.update({f"{param_name}/{k}": v for k, v in nested_flat_params.items()})
            else:
                flat_params[param_name] = param
        return flat_params


class DesignParameters:

    panel_data = None

    def __init__(self, path=None, source_dict=None, dataset='gcd'):
        self.dataset = dataset
        if source_dict is not None:
            self.design_parameters = source_dict
            self.path = None
        else:
            self.design_parameters = self._load_yaml(path)
            self.path = path
        
        self._setup_panel_data()
        self.panel_data = self.flatten()

    def _load_yaml(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_panel_data(self):
        panel_data = {}
        for key, values in self.design_parameters['design'].items():
            panel_data[key] = PanelData(key, values)
        self.panel_data = panel_data
    
    def flatten(self):
        flat_params = {}
        for key, panel in self.panel_data.items():
            flat_params[key] = panel.flatten()
        return flat_params

    def get_design_parameters(self):
        return self.design_parameters
    
    def get_panel_data(self):
        return self.panel_data
    
    def aggregate_results(self, evaluation_results):
        loss = 0.
        for key, value in evaluation_results.items():
            if isinstance(value, dict) and "loss" in value:
                loss += value["loss"]
            elif isinstance(value, dict):
                loss += self.aggregate_results(value)
            else:
                continue
        return loss / len(evaluation_results.keys())
    
    def __getitem__(self, key):
        return self.panel_data[key]
    
    def keys(self):
        return self.panel_data.keys()
    
    def create_gt_from_file(self, path):
        # clone this instance, and set all the values from the file.
        # this has to be done since the parameters of the DesignPattern in the file may not match the ones in this instance.
        this_copy = copy.deepcopy(self)
        other_design_parameters = DesignParameters(path)
        this_copy.set_values(other_design_parameters)
        return this_copy

    def create_gt_from_dict(self, design_inner):
        # Like create_gt_from_file but from an in-memory design dict (the inner
        # 'design' mapping). Used to build a GT for a merged top+bottom outfit.
        this_copy = copy.deepcopy(self)
        other_design_parameters = DesignParameters(source_dict={'design': design_inner})
        this_copy.set_values(other_design_parameters)
        return this_copy

    def crate_gt_from_specifications(self, specifications_path):
        yaml_dict = load_spec_to_yaml_format(specifications_path, default_yaml)
        
    
    def set_values(self, other_design_parameters):
        for panel, parameters_dict in self.get_panel_data().items():
            for param_name, param in parameters_dict.items():
                    try:
                        other_param = other_design_parameters[panel][param_name]
                    except (KeyError, TypeError):
                        # Param/panel absent in the source design yaml. Happens for
                        # GCD garments that predate the RC params (shirt.openfront,
                        # waistband.height) and for the composition-only label
                        # meta.is_two_piece. Keep this (template) instance's default.
                        continue
                    self.panel_data[panel][param_name].set_value(param.value_of(other_param))

    def evaluate(self, other):
        evaluation_results = {}
        param: DefaultParameter
        for panel_name, panel_params in self.panel_data.items():
            evaluation_results[panel_name] = {}
            for param_name, param in panel_params.items():
                loss = param.loss(other[panel_name][param_name])
                evaluation_results[panel_name][param_name] = {
                    'loss': loss,
                    'prediction': other[panel_name][param_name],
                    'gt': param.get_value(),
                    'type': param.type,
                }
        return evaluation_results


    def perturb_sample(self):
        for panel_name, panel_params in self.panel_data.items():
            for param_name, param in panel_params.items():
                if param.type == 'float':
                    perturb = torch.rand((1,), device=param.value.device) * (param.max_value - param.min_value)
                    print(f"{panel_name}/{param_name}: {param.value} -> {max(param.min_value, min(param.max_value, param.value + perturb))}   -   range [{param.min_value}, {param.max_value}]")
                    param.value = (param.value + perturb).clip(param.min_value, param.max_value)

    def edit_design(self, noise_scale=0.1):
        """
        Perturbs the active float parameters of the design.
        """
        # 1. Create a dictionary of current values acting as 'predictions'
        predictions = {}
        for panel_name, panel_params in self.panel_data.items():
            predictions[panel_name] = {}
            for param_name, param in panel_params.items():
                predictions[panel_name][param_name] = param.get_value()

        # 2. Get active mask
        active_mask = self.get_active_mask(predictions)

        # 3. Perturb active float parameters
        for panel_name, panel_params in self.panel_data.items():
            for param_name, param in panel_params.items():
                is_active = active_mask.get(panel_name, {}).get(param_name, False)
                
                if is_active and param.type == 'float':
                    # range of the parameter
                    param_range = param.max_value - param.min_value
                    
                    # Perturbation: random value in [-0.5, 0.5] * range * scale
                    perturb = (torch.rand((1,), device=param.value.device) - 0.5) * param_range * noise_scale
                    
                    new_value = (param.value + perturb).clip(param.min_value, param.max_value)
                    
                    # print(f"Editing {panel_name}/{param_name}: {param.value.item():.4f} -> {new_value.item():.4f}")
                    param.value = new_value


    def to_dict(self):
        # copy this design param in a dict for export.
        if self.path is not None:
            d = self._load_yaml(self.path)
        else:
            d = copy.deepcopy(self.design_parameters)

        for panel_name, panel_params in self.panel_data.items():
            for param_name, param in panel_params.items():
                names = param_name.split('/')[:-1]
                reduce(operator.getitem, names, d['design'][panel_name])[param_name.split('/')[-1]] = param.serialize()

        return d


    def evaluate_geometry(
        self,
        predicted_pattern_svg,
        gt_pattern_svg,
        samples_per_path=200,
        device="cuda",
    ):
        """
        Computes Chamfer distance between two SVGs by sampling points from their paths.

        Args:
            predicted_pattern_svg (str): path to predicted SVG file
            gt_pattern_svg (str): path to GT SVG file
            samples_per_path (int): number of points sampled per SVG path
            normalize (bool): whether to normalize point clouds to unit box
            device (str): torch device

        Returns:
            torch.Tensor: scalar Chamfer distance
        """
        from svgpathtools import svg2paths
        import torch

        def chamfer_distance(p1, p2):
            """
            Symmetric Chamfer distance.
            """
            if p1.numel() == 0 or p2.numel() == 0:
                return torch.tensor(0.0, device=p1.device)

            dists = torch.cdist(p1, p2)  # (N, M)
            return dists.min(dim=1)[0].mean() + dists.min(dim=0)[0].mean()

        def sample_svg_points(svg_path, samples_per_path):
            """
            Samples points uniformly along arc length for each path.
            """
            paths, _ = svg2paths(svg_path)
            points = []

            for path in paths:
                length = path.length()
                if length <= 0:
                    continue

                # Uniform sampling in arc-length space
                for i in range(samples_per_path):
                    t = path.ilength(length * (i / samples_per_path))
                    p = path.point(t)
                    points.append([p.real, p.imag])

            if len(points) == 0:
                return torch.empty((0, 2), dtype=torch.float32)

            return torch.tensor(points, dtype=torch.float32)

        # def normalize_points(pts):
        #     """
        #     Normalizes points to zero mean and unit bounding box.
        #     """
        #     min_xy = pts.min(dim=0)[0]
        #     max_xy = pts.max(dim=0)[0]
        #     scale = (max_xy - min_xy).max()
        #     return (pts - (min_xy + max_xy) / 2) / (scale + 1e-8)

        # --- Load & sample ---
        pts_pred = sample_svg_points(predicted_pattern_svg, samples_per_path)
        pts_gt = sample_svg_points(gt_pattern_svg, samples_per_path)

        import matplotlib.pyplot as plt
        plt.scatter(pts_pred[:, 0], pts_pred[:, 1], c='r')
        plt.scatter(pts_gt[:, 0], pts_gt[:, 1], c='b')
        plt.show()

        pts_pred = pts_pred.to(device)
        pts_gt = pts_gt.to(device)

        # --- Optional normalization ---
        # if normalize and pts_pred.numel() > 0 and pts_gt.numel() > 0:
        #     pts_pred = normalize_points(pts_pred)
        #     pts_gt = normalize_points(pts_gt)

        # --- Chamfer distance ---
        return chamfer_distance(pts_pred, pts_gt)

        


    def to(self, device):
        for panel_key in self.panel_data.keys():
            for param_keys in self.panel_data[panel_key].keys():
                self.panel_data[panel_key][param_keys] = self.panel_data[panel_key][param_keys].to(device)
        return self
    
    def prediction_to_value(self, prediction, panel_name, parameter_name, differentiable=False):
        return self.panel_data[panel_name][parameter_name].prediction_to_value(prediction, differentiable=differentiable)
    
    def get_value(self, panel_name, parameter_name):
        return self.panel_data[panel_name][parameter_name]

    def _prediction_to_yaml(self, predictions, differentiable=False):
        yaml_prediction = copy.deepcopy(self.get_design_parameters())

        for panel_name, panel_values in predictions.items():
            for parameter_name, parameter_value in panel_values.items():
                splitted_param_name = parameter_name.split("/")
                _parameter_value = self.prediction_to_value(prediction=parameter_value, panel_name=panel_name, parameter_name=parameter_name, differentiable=differentiable)
                setInDict(yaml_prediction, ["design", panel_name, *splitted_param_name, "v"], _parameter_value)
        return yaml_prediction

    def _prediction_to_multiple_yaml(self, predictions, differentiable=False):
        "same as the previous one, but return a separate yaml for every prediction."
        yaml_prediction = []
        # copy.deepcopy(self.get_design_parameters())
        B = len(predictions['meta']['upper'])

        # Create B copies of the design parameters
        for i in range(B):
            yaml_prediction.append(copy.deepcopy(self.get_design_parameters()))

        # for each value, copy the corresponding value in the yaml_prediction list
        for panel_name, panel_values in predictions.items():
            for parameter_name, parameter_value in panel_values.items():
                splitted_param_name = parameter_name.split("/")
                _parameter_value = self.prediction_to_value(prediction=parameter_value, panel_name=panel_name, parameter_name=parameter_name, differentiable=differentiable)
                for i in range(B):
                    setInDict(yaml_prediction[i], ["design", panel_name, *splitted_param_name, "v"], _parameter_value[i])
        return yaml_prediction

    def export_prediction_to_yaml(self, predictions, out_path, test_files):
        # extract predictions for each sample of the batch, and save them
        yaml_prediction = self._prediction_to_yaml(predictions)

        self._save_batched_yaml_to_file(yaml_prediction, out_path, test_files)

    def _unbatch_prediction(self, batched_yaml_prediction):
        batch_size = len(batched_yaml_prediction["design"]["meta"]["upper"]["v"])
        yaml_predictions_list = [batched_yaml_prediction for i in range(batch_size)] # prepare a yaml for each element in the batch
        # yaml_predictions_list = [copy.deepcopy(batched_yaml_prediction) for i in range(batch_size)] # prepare a yaml for each element in the batch

        for idx in range(batch_size):
            self._copy_dict_to(batched_yaml_prediction, yaml_predictions_list[idx], idx)
        
        return yaml_predictions_list

    def _save_batched_yaml_to_file(self, yaml_prediction, out_path, test_files):
        # batch_size = len(yaml_prediction["design"]["meta"]["upper"]["v"])
        
        # yaml_predictions_list = [copy.deepcopy(yaml_prediction) for i in range(batch_size)] # prepare a yaml for each element in the batch

        # for idx in range(batch_size):
        #     self._copy_dict_to(yaml_prediction, yaml_predictions_list[idx], idx)
        yaml_predictions_list = self._unbatch_prediction(yaml_prediction)
        for idx, _yaml in enumerate(yaml_predictions_list):
            self.to_specification_file(_yaml["design"], out_path=out_path, sample_name=test_files[idx])

        
    def _copy_dict_to(self, source, target, idx):
        for key in source.keys():
            if isinstance(source[key], dict):
                self._copy_dict_to(source[key], target[key], idx)
            else:
                if key == "v":
                    target[key] = source[key][idx]

    def build_metagarment(self, _yaml, sample_name):
        """
        Avoid this function
        """
        try:
            new_design = _yaml['design']
            name = f'prediction_{sample_name}'
            default_body = BodyParameters(Path("../GarmentCode/assets/bodies/mean_all.yaml"))

            # On default body
            piece_default = MetaGarment(name, default_body, new_design, device="cpu") 

            # Straight/apart legs pose
            def_obj_name = "../GarmentCode/assets/bodies/mean_all"

            if has_pants(new_design):
                def_obj_name += '_apart'
            default_body.params['body_sample'] = def_obj_name + ".obj"
        except Exception as e:
            # e.with_traceback()
            print(f"Failed to extract {sample_name} - {repr(e)}")

        # except IncorrectElementConfiguration as e:
        #     print(f"Failed to extract {sample_name} - {repr(e)}")

        return default_body, piece_default

    def build_differentiable_metagarment(self, _yaml, sample_name, device):
        # try:
        new_design = _yaml['design']
        name = f'prediction_{sample_name}'
        default_body = BodyParameters(Path("../GarmentCode/assets/bodies/mean_all.yaml"), device=device)

        # On default body
        piece = DifferentiableMetaGarment(name, default_body, new_design, device=device)
        # pattern = piece.assembly()
        # garment_box_mesh = BoxMesh(pattern_spec=pattern.spec, res=1.)
        # garment_box_mesh.load()

        # Straight/apart legs pose
        def_obj_name = "../GarmentCode/assets/bodies/mean_all"

        if has_pants(new_design):
            def_obj_name += '_apart'
        default_body.params['body_sample'] = def_obj_name + ".obj"

        return default_body, piece

    def to_specification_file(self, new_design, out_path, sample_name, device="cpu"):
        default_body, piece_default = self.build_differentiable_metagarment(new_design, sample_name=sample_name, device=device)
        # Save samples
        pattern = _save_sample(piece_default, default_body, new_design, out_path, verbose=False)


    def get_boxmesh(self, predictions, sample_name):
        yaml_prediction = self._prediction_to_yaml(predictions)
        unbatched = self._unbatch_prediction(yaml_prediction)
        boxmesh_list = []
        for i in range(len(sample_name)):
            # try:
                default_body, piece_default = self.build_metagarment(unbatched[i], sample_name=sample_name[i])
                pattern = piece_default.assembly()
                from src.external.GarmentCode.pygarment.meshgen.boxmeshgen import BoxMesh  # optional helper (needs igl + CGAL)
                garment_box_mesh = BoxMesh(None, spec=pattern.spec, res=1.)
                garment_box_mesh.load()
                pts = garment_box_mesh.vertices
                boxmesh_list.append(pts)
            # except Exception as e:
            #     print(f"Error while extracting {sample_name[i]}. Error: {e}")
            # except DegenerateTrianglesError as e:
            #     print(f"Error while extracting {sample_name[i]}. Error: {e}")
        return boxmesh_list
    

    def get_differentiable_boxmesh(self, predictions, sample_name, device):
        differentiable_yaml_prediction = self._prediction_to_yaml(predictions, differentiable=True)
        unbatched = self._unbatch_prediction(differentiable_yaml_prediction)
        boxmesh_list = []
        for i in range(len(sample_name)):
            # try:
                pts = self.build_differentiable_metagarment(unbatched[i], sample_name=sample_name[i], device=device)
                boxmesh_list.append(pts)
            # except Exception as e:
            #     print(f"Error while extracting {sample_name[i]}. Error: {e}")
            # except DegenerateTrianglesError as e:
            #     print(f"Error while extracting {sample_name[i]}. Error: {e}")
        return boxmesh_list
            


    def get_active_mask(self, predictions, return_dict=False):
        """
        Determine which parameters are active based on discrete predictions.
        Returns a mask dictionary with same structure as predictions.
        
        Args:
            predictions: dict of predicted parameter values
            return_dict: if True, returns dict; if False, returns boolean values
        
        Returns:
            active_mask: dict indicating which parameters should contribute to loss
        """
        active_mask = {}
        
        # Decode meta predictions (always active as they determine structure)
        upper_type = self.prediction_to_value(
            predictions['meta']['upper'], 'meta', 'upper')[0]
        wb_type = self.prediction_to_value(
            predictions['meta']['wb'], 'meta', 'wb')[0]
        bottom_type = self.prediction_to_value(
            predictions['meta']['bottom'], 'meta', 'bottom')[0]
        
        # Meta parameters - always active (structure determiners)
        active_mask['meta'] = {
            param: True for param in predictions['meta'].keys()
        }
        # ...except the layering label, which is meaningless for one-piece
        # garments: only supervise it when the outfit is actually two-piece.
        if 'top_tucked' in predictions['meta'] and 'is_two_piece' in predictions['meta']:
            is_two = self.prediction_to_value(
                predictions['meta']['is_two_piece'], 'meta', 'is_two_piece')[0]
            active_mask['meta']['top_tucked'] = bool(is_two)
            # ...and the top's own waistband selector: only a separate top can
            # carry a second waistband (dress/one-piece has just meta.wb).
            if 'top_wb' in predictions['meta']:
                active_mask['meta']['top_wb'] = bool(is_two)

        # Waistband parameters - active if wb_type is not None
        active_mask['waistband'] = {
            param: (wb_type is not None)
            for param in predictions['waistband'].keys()
        }

        # Top garment's waistband (two-piece outfits) - active if its own
        # selector meta.top_wb is not None
        if 'top_waistband' in predictions:
            top_wb_type = self.prediction_to_value(
                predictions['meta']['top_wb'], 'meta', 'top_wb')[0] \
                if 'top_wb' in predictions['meta'] else None
            active_mask['top_waistband'] = {
                param: (top_wb_type is not None)
                for param in predictions['top_waistband'].keys()
            }
        
        # Upper garment parameters - active if upper_type is not None
        active_mask['shirt'] = {
            param: (upper_type is not None)
            for param in predictions['shirt'].keys()
        }
        
        active_mask['sleeve'] = {
            param: (upper_type is not None)
            for param in predictions['sleeve'].keys()
        }
        
        # Left asymmetry parameters - refined check for enable_asym
        active_mask['left'] = {}
        for param in predictions['left'].keys():
            if param == 'enable_asym':
                # Always active when upper exists (it's the control parameter)
                active_mask['left'][param] = (upper_type is not None)
            else:
                # Other left params only matter if asymmetry is enabled
                if upper_type is not None:
                    enable_asym = self.prediction_to_value(
                        predictions['left']['enable_asym'], 'left', 'enable_asym')[0]
                    active_mask['left'][param] = (enable_asym is True)
                else:
                    active_mask['left'][param] = False
        
        # Collar parameters - active if upper_type is not None
        active_mask['collar'] = {}
        for param in predictions['collar'].keys():
            if upper_type is None:
                active_mask['collar'][param] = False
            elif param.startswith('component/'):
                # Component style is always active when upper exists
                if param == 'component/style':
                    active_mask['collar'][param] = True
                else:
                    # Other component params only active if style is not None
                    collar_style = self.prediction_to_value(
                        predictions['collar']['component/style'], 
                        'collar', 'component/style')[0]
                    active_mask['collar'][param] = (collar_style is not None)
            else:
                active_mask['collar'][param] = True
        
        # Handle sleeve cuff - active if upper exists and not sleeveless
        if upper_type is not None and 'cuff/type' in predictions['sleeve']:
            sleeveless = self.prediction_to_value(
                predictions['sleeve']['sleeveless'], 'sleeve', 'sleeveless')[0]
            cuff_type = self.prediction_to_value(
                predictions['sleeve']['cuff/type'], 'sleeve', 'cuff/type')[0]

            for param in predictions['sleeve'].keys():
                if param.startswith('cuff/'):
                    if param == 'cuff/type':
                        active_mask['sleeve'][param] = (not sleeveless)
                    else:
                        active_mask['sleeve'][param] = (not sleeveless and cuff_type is not None)

        # Cut parameters - for slits in circle skirts
        if 'cut' in predictions:
            active_mask['cut'] = {}
            for param in predictions['cut'].keys():
                # Cut parameters only relevant for circle skirts
                active_mask['cut'][param] = bottom_type in ['SkirtCircle', 'AsymmSkirtCircle']

        # Bottom parameters - complex dependencies
        active_mask.update(self._get_bottom_active_mask(predictions, bottom_type))
        
        return active_mask
    
    def _get_bottom_active_mask(self, predictions, bottom_type):
        """Helper to determine active bottom parameters with dependencies"""
        mask = {}
        
        # Initialize all bottom sections as inactive
        for section in ['skirt', 'flare-skirt', 'godet-skirt', 
                        'pencil-skirt', 'levels-skirt', 'pants']:
            mask[section] = {param: False for param in predictions[section].keys()}
        
        if bottom_type is None:
            return mask
        
        # Activate based on bottom type
        if bottom_type in ['SkirtCircle', 'AsymmSkirtCircle']:
            # Uses flare-skirt parameters
            for param in predictions['flare-skirt'].keys():
                mask['flare-skirt'][param] = True
                
        elif bottom_type == 'SkirtManyPanels':
            # Uses flare-skirt and specific sub-params
            for param in predictions['flare-skirt'].keys():
                if param.startswith('skirt-many-panels/') or param in ['suns', 'length', 'rise']:
                    mask['flare-skirt'][param] = True
                    
        elif bottom_type == 'Skirt2':
            for param in predictions['skirt'].keys():
                mask['skirt'][param] = True
                
        elif bottom_type == 'PencilSkirt':
            for param in predictions['pencil-skirt'].keys():
                mask['pencil-skirt'][param] = True
                
        elif bottom_type == 'GodetSkirt':
            # Godet uses its own params plus base skirt params
            for param in predictions['godet-skirt'].keys():
                mask['godet-skirt'][param] = True
            
            # Determine base skirt and activate its params
            base_type = self.prediction_to_value(
                predictions['godet-skirt']['base'], 'godet-skirt', 'base')[0]
            
            if base_type == 'Skirt2':
                for param in predictions['skirt'].keys():
                    mask['skirt'][param] = True
            elif base_type == 'PencilSkirt':
                for param in predictions['pencil-skirt'].keys():
                    mask['pencil-skirt'][param] = True
                    
        elif bottom_type == 'SkirtLevels':
            # Levels uses its own params plus base and level skirt params
            for param in predictions['levels-skirt'].keys():
                mask['levels-skirt'][param] = True
            
            # Determine base and level skirts
            base_type = self.prediction_to_value(
                predictions['levels-skirt']['base'], 'levels-skirt', 'base')[0]
            level_type = self.prediction_to_value(
                predictions['levels-skirt']['level'], 'levels-skirt', 'level')[0]
            
            # Activate base skirt params
            if base_type in ['SkirtCircle', 'AsymmSkirtCircle']:
                for param in predictions['flare-skirt'].keys():
                    mask['flare-skirt'][param] = True
            elif base_type == 'Skirt2':
                for param in predictions['skirt'].keys():
                    mask['skirt'][param] = True
            elif base_type == 'PencilSkirt':
                for param in predictions['pencil-skirt'].keys():
                    mask['pencil-skirt'][param] = True
            
            # Activate level skirt params
            if level_type in ['SkirtCircle', 'AsymmSkirtCircle']:
                for param in predictions['flare-skirt'].keys():
                    mask['flare-skirt'][param] = True
            elif level_type == 'Skirt2':
                for param in predictions['skirt'].keys():
                    mask['skirt'][param] = True
                    
        elif bottom_type == 'Pants':
            for param in predictions['pants'].keys():
                mask['pants'][param] = True
            
            # Handle pants cuff
            if 'cuff/type' in predictions['pants']:
                cuff_type = self.prediction_to_value(
                    predictions['pants']['cuff/type'], 'pants', 'cuff/type')[0]
                for param in predictions['pants'].keys():
                    if param.startswith('cuff/') and param != 'cuff/type':
                        mask['pants'][param] = (cuff_type is not None)
        
        return mask
    
    def evaluate_with_mask(self, other):
        """
        Evaluate predictions with activity masking.
        Discrete parameters always contribute to loss.
        Continuous parameters only contribute if active.
        """
        evaluation_results = {}
        active_mask = self.get_active_mask(other)
        
        for panel_name, panel_params in self.panel_data.items():
            evaluation_results[panel_name] = {}
            
            for param_name, param in panel_params.items():
                # Compute loss
                loss = param.loss(other[panel_name][param_name])
                
                # Determine if this parameter should contribute to backprop
                is_active = active_mask.get(panel_name, {}).get(param_name, True)
                is_discrete = param.type in ['select', 'select_null', 'bool']
                
                # Discrete params always backprop, continuous only if active
                should_backprop = is_discrete or is_active
                
                # Store detached loss if not backpropping
                if not should_backprop:
                    loss = loss.detach()
                
                evaluation_results[panel_name][param_name] = {
                    'loss': loss,
                    'prediction': other[panel_name][param_name],
                    'gt': param.get_value(),
                    'type': param.type,
                    'active': is_active,
                    'backprop': should_backprop
                }
        
        return evaluation_results
    
    # def aggregate_results_with_mask(self, evaluation_results):
    #     """Aggregate only backprop-enabled losses"""
    #     total_loss = 0.
    #     count = 0
        
    #     for panel_name, panel_values in evaluation_results.items():
    #         for param_name, param_info in panel_values.items():
    #             if isinstance(param_info, dict) and 'loss' in param_info:
    #                 if param_info.get('backprop', True):  # Only aggregate if should backprop
    #                     total_loss += param_info['loss']
    #                     count += 1
        
    #     return total_loss / max(count, 1)  # Avoid division by zero

    def aggregate_results_with_mask(self, evaluation_results):
        """
        Instead of averaging everything together, normalize each loss 
        type separately before combining.
        """
        losses = {
            'select': [],
            'select_null': [],
            'bool': [],
            'float': [],
            'int': [],
        }

        for panel_name, panel_values in evaluation_results.items():
            for param_name, param_info in panel_values.items():
                if not isinstance(param_info, dict) or 'loss' not in param_info:
                    continue
                if not param_info.get('backprop', True):
                    continue

                param_type = param_info['type']
                if param_type in losses:
                    losses[param_type].append(param_info['loss'])

        # Average within each type
        type_losses = {}
        for loss_type, loss_list in losses.items():
            if loss_list:
                type_losses[loss_type] = torch.stack(loss_list).mean()

        return type_losses

    def to_input_dict(self):
        """
        Returns the parameter values as a nested dict
        {panel_name -> {param_name -> tensor [B, size]}}
        ready to be fed into PatternVAE.encode().
        """
        out = {}
        for panel_name, panel_params in self.panel_data.items():
            out[panel_name] = {}
            for param_name, param in panel_params.items():
                out[panel_name][param_name] = param.get_value()  # already normalised
        return out


class ClassificationHeads(torch.nn.Module):
    def __init__(self, design_params: DesignParameters, input_dim):
        super(ClassificationHeads, self).__init__()
        self.design_params = design_params
        self.input_dim = input_dim

    
class ParametersClassificationHeads(ClassificationHeads):
    
    def __init__(self, design_params: DesignParameters, input_dim):
        super(ParametersClassificationHeads, self).__init__(design_params, input_dim)
        self.heads = None
        self._create_heads()
    
    def _create_heads(self):
        self.heads = torch.nn.ModuleDict()
        # flat_params = self.design_params.flatten()
        for panel_name, panel_params in self.design_params.get_panel_data().items():
            self.heads[panel_name] = torch.nn.ModuleDict()
            for param_name, param in panel_params.items():
                param_class = type(param)
                head_class = PARAMETER_TO_HEAD_MAP.get(param_class, None)
                if head_class is not None:
                    output_size = param.get_output_size()
                    head_instance = head_class(self.input_dim, output_size, **param.get_options())
                    self.heads[panel_name][param_name] = head_instance
                else:
                    print(f"Warning: No head defined for parameter class '{param_class}'")
    
    def forward(self, x):
        outputs = {}
        for panel_name, panel_heads in self.heads.items():
            outputs[panel_name] = {}
            for param_name, head in panel_heads.items():
                outputs[panel_name][param_name] = head(x)
        return outputs









##############################################################################




class RangeRegressionHeadV2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, min_val=0, max_val=1, dropout=0.3, **kwargs):
        super(RangeRegressionHeadV2, self).__init__()
        # Simpler architecture
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 4)
        self.norm1 = torch.nn.LayerNorm(input_dim // 4)
        self.dropout1 = torch.nn.Dropout(dropout)
        
        self.fc2 = torch.nn.Linear(input_dim // 4, output_dim)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.min_val = min_val
        self.max_val = max_val
        self.linear_only = kwargs.get('linear_only', False)

        if self.linear_only:
            self.fc1 = torch.nn.Linear(input_dim, output_dim, bias=False)
            self.norm1 = torch.nn.Identity()
            self.dropout1 = torch.nn.Identity()
            self.fc2 = torch.nn.Identity()
            self.relu = torch.nn.Identity()
            self.sigmoid = torch.nn.Identity()

        
        # Weight initialization - important!
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MulticlassRegressionHeadV2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, **kwargs):
        super(MulticlassRegressionHeadV2, self).__init__()
        self.num_classes = output_dim
        
        # Even simpler for classification
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 4)
        self.norm1 = torch.nn.LayerNorm(input_dim // 4)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(input_dim // 4, self.num_classes)
        self.relu = torch.nn.ReLU()
        self.linear_only = kwargs.get('linear_only', False)

        if self.linear_only:
            self.fc1 = torch.nn.Linear(input_dim, self.num_classes, bias=False)
            self.norm1 = torch.nn.Identity()
            self.dropout1 = torch.nn.Identity()
            self.fc2 = torch.nn.Identity()
            self.relu = torch.nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x  # Return logits, not softmax


class BinaryRegressionHeadV2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, **kwargs):
        super(BinaryRegressionHeadV2, self).__init__()
        # Single layer is often enough for binary
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_only = kwargs.get('linear_only', False)

        if self.linear_only:
            self.fc = torch.nn.Linear(input_dim, output_dim, bias=False)
            self.dropout = torch.nn.Identity()
            self.sigmoid = torch.nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x






PARAMETER_TO_HEAD_MAP_V2 = {
    FloatParameter: RangeRegressionHeadV2,
    IntParameter: RangeRegressionHeadV2,
    BoolParameter: BinaryRegressionHeadV2,
    SelectParameter: MulticlassRegressionHeadV2,
    SelectNullParameter: MulticlassRegressionHeadV2
}




class ParametersClassificationHeadsWithSharedFeatures(ClassificationHeads):
    
    def __init__(self, design_params: DesignParameters, input_dim, dropout=0.3, div=1, linear_only=False):
        super(ParametersClassificationHeadsWithSharedFeatures, self).__init__(design_params, input_dim)
        self.dropout = dropout
        self.div = div
        self.linear_only = linear_only
        
        if self.linear_only:
            self.shared_features = torch.nn.Identity()
            self.div = 1 # Dimension shouldn't be divided if linear
        else:
            # Shared feature extractor with dropout
            self.shared_features = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // div),
                torch.nn.LayerNorm(input_dim // div),  # Add normalization
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
        
        self.heads = torch.nn.ModuleDict()
        self._create_heads()
    
    def _create_heads(self):
        # Use shared features dimension
        shared_dim = self.input_dim // self.div
        
        for panel_name, panel_params in self.design_params.get_panel_data().items():
            self.heads[panel_name] = torch.nn.ModuleDict()
            for param_name, param in panel_params.items():
                param_class = type(param)
                head_class = PARAMETER_TO_HEAD_MAP_V2.get(param_class, None)
                if head_class is not None:
                    output_size = param.get_output_size()
                    head_instance = head_class(
                        shared_dim,  # Use shared features
                        output_size, 
                        dropout=self.dropout,  # Pass dropout to heads
                        linear_only=self.linear_only, # Enforce linearity constraint locally inside headers
                        **param.get_options()
                    )
                    self.heads[panel_name][param_name] = head_instance
    
    def forward(self, x):
        # Extract shared features once
        shared_feat = self.shared_features(x)
        
        outputs = {}
        for panel_name, panel_heads in self.heads.items():
            outputs[panel_name] = {}
            for param_name, head in panel_heads.items():
                outputs[panel_name][param_name] = head(shared_feat)
        return outputs
    
def collate_parameters(batch):
    B = len(batch)
    collated = batch[0]
    panel_keys = batch[0].keys()
    for panel_name in panel_keys:
        param_keys = batch[0][panel_name].keys()
        for param_name in param_keys:
            collated[panel_name][param_name].join([el[panel_name][param_name] for el in batch[1:]])
    return collated


if __name__ == "__main__":

    test_files = [
        "rand_00YONAPXZE",
        "rand_2M8JA3LPWR",
        "rand_GR0Y96XSTC"
    ]

    batch_size = len(test_files)

    targets = [f"../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2/garments_5000_0/default_body/data/{f}/{f}_design_params.yaml" for f in test_files]
    template_path = "../GarmentCode/assets/design_params/default.yaml"

    design_parameters_template = DesignParameters(template_path)
    
    design_parameters_classifier_heads = ParametersClassificationHeads(design_parameters_template, input_dim=128)
    design_parameters_classifier_heads.train()

    gt_parameters = collate_parameters([
        design_parameters_template.create_gt_from_file(target_file) for target_file in targets
        ])

    input_tensor = torch.randn((batch_size, 128))    # dummy latent for the garment
    input_tensor = input_tensor.float().requires_grad_(False)

    predicted_parameters = design_parameters_classifier_heads(input_tensor)
    # error = gt_parameters.evaluate(predicted_parameters)

    optimizer = torch.optim.Adam(design_parameters_classifier_heads.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    num_steps = 100
    pbar = tqdm.tqdm(range(num_steps), total=num_steps)
    for i in pbar:
        optimizer.zero_grad()
        predicted_parameters = design_parameters_classifier_heads(input_tensor)

        loss_dict = gt_parameters.evaluate(predicted_parameters)
        loss = gt_parameters.aggregate_results(loss_dict)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
    boxmesh_list = design_parameters_template.get_boxmesh(predicted_parameters, test_files)
    import trimesh
    for idx, el in enumerate(boxmesh_list):
        el = trimesh.Trimesh(el[0], el[1], process=False)
        el.export(f"tmp/extracted_boxmesh_{idx}.ply")
    exit()


# ============================================================================
# V3 Heads — No sigmoid for range regression, better gradient flow
# ============================================================================

class RangeRegressionHeadV3(torch.nn.Module):
    """
    Predicts continuous values WITHOUT sigmoid.
    The network outputs raw [0,1]-normalized values directly,
    supervised by MSE against normalized GT. This avoids the
    gradient saturation problem of sigmoid at extremes.
    """
    def __init__(self, input_dim, output_dim, min_val=0, max_val=1, dropout=0.1, **kwargs):
        super().__init__()
        hidden = max(input_dim // 4, 32)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, output_dim),
        )
        self.min_val = min_val
        self.max_val = max_val
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.5)  # bias towards midpoint

    def forward(self, x):
        return self.net(x)  # raw output, no sigmoid


class BinaryRegressionHeadV3(torch.nn.Module):
    """Binary classification — sigmoid is kept since BCE loss requires it."""
    def __init__(self, input_dim, output_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, max(input_dim // 4, 16)),
            torch.nn.LayerNorm(max(input_dim // 4, 16)),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(max(input_dim // 4, 16), output_dim),
            torch.nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class MulticlassRegressionHeadV3(torch.nn.Module):
    """Multiclass classification — outputs raw logits for CrossEntropyLoss."""
    def __init__(self, input_dim, output_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.num_classes = output_dim
        hidden = max(input_dim // 4, 32)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, self.num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)  # raw logits


PARAMETER_TO_HEAD_MAP_V3 = {
    FloatParameter: RangeRegressionHeadV3,
    IntParameter: RangeRegressionHeadV3,      # int treated as continuous
    BoolParameter: BinaryRegressionHeadV3,
    SelectParameter: MulticlassRegressionHeadV3,
    SelectNullParameter: MulticlassRegressionHeadV3
}

HEAD_VERSION_MAPS = {
    'v1': PARAMETER_TO_HEAD_MAP,
    'v2': PARAMETER_TO_HEAD_MAP_V2,
    'v3': PARAMETER_TO_HEAD_MAP_V3,
}


import torch.nn.functional as F


class PanelGroupedAttentionHeads(ClassificationHeads):
    """
    Panel-grouped cross-attention heads for compositional generalization.

    Each physical panel (e.g. 'sleeve_lf', 'skirt_front') gets its own
    learnable query vector that cross-attends over the full latent sequence.
    All parameters of that panel — including its meta existence flag
    (meta.sleeve_lf) — are predicted from the **same** panel-specific
    pooled feature.

    Key benefit: since each panel's existence and shape parameters are
    grounded in its own geometric region, the model can predict unseen
    combinations at test time (e.g. tee sleeves + dress skirt) by
    independently detecting each part.

    Input:  x  (B, N, D) — sequence of latent tokens from the encoder.
    Output: {panel_name -> {param_name -> (B, out_dim)}} and 'panel_features' for RDM loss
    """

    def __init__(
        self,
        design_params: DesignParameters,
        input_dim: int,
        dropout: float = 0.1,
        head_version: str = 'v3',
        num_attention_heads: int = 8,
    ):
        super().__init__(design_params, input_dim)
        self.dropout = dropout
        self.head_map = HEAD_VERSION_MAPS.get(head_version, PARAMETER_TO_HEAD_MAP_V3)
        self.num_heads = num_attention_heads
        assert input_dim % num_attention_heads == 0, \
            f"input_dim {input_dim} must be divisible by num_attention_heads {num_attention_heads}"
        self.head_dim = input_dim // num_attention_heads

        # Physical panels (everything except 'meta')
        panel_data = design_params.get_panel_data()
        self.non_meta_panels = [k for k in panel_data.keys() if k != 'meta']
        self.panel_to_idx = {name: i for i, name in enumerate(self.non_meta_panels)}

        # One learnable query per physical panel: (P, D)
        num_panels = len(self.non_meta_panels)
        self.panel_queries = torch.nn.Parameter(
            torch.randn(num_panels, input_dim) * 0.02
        )

        # Shared K/V projection over the full latent sequence
        self.to_kv = torch.nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.attn_out = torch.nn.Linear(input_dim, input_dim)
        self.norm = torch.nn.LayerNorm(input_dim)

        self.heads = torch.nn.ModuleDict()
        self._create_heads()

    def _create_heads(self):
        panel_data = self.design_params.get_panel_data()

        # Meta panel: existence heads (one per physical panel)
        if 'meta' in panel_data:
            self.heads['meta'] = torch.nn.ModuleDict()
            for param_name, param in panel_data['meta'].items():
                head_class = self.head_map.get(type(param))
                if head_class is None:
                    raise ValueError(f"No head for '{type(param)}'")
                self.heads['meta'][param_name] = head_class(
                    self.input_dim, param.get_output_size(),
                    dropout=self.dropout, **param.get_options()
                )

        # Physical panels: geometry heads
        for panel_name in self.non_meta_panels:
            if panel_name not in panel_data:
                continue
            self.heads[panel_name] = torch.nn.ModuleDict()
            for param_name, param in panel_data[panel_name].items():
                head_class = self.head_map.get(type(param))
                if head_class is None:
                    raise ValueError(f"No head for '{type(param)}'")
                self.heads[panel_name][param_name] = head_class(
                    self.input_dim, param.get_output_size(),
                    dropout=self.dropout, **param.get_options()
                )

    def _compute_panel_features(self, x):
        """Cross-attend each panel query over the full latent sequence.
        Returns (B, P, D) panel feature matrix."""
        B, N, D = x.shape
        P = len(self.non_meta_panels)
        H, d = self.num_heads, self.head_dim

        # Panel queries: (B, P, D)
        q = self.panel_queries.unsqueeze(0).expand(B, -1, -1)

        # Keys and values from the full latent sequence
        kv = self.to_kv(x)          # (B, N, 2D)
        k, v = kv.chunk(2, dim=-1)  # each (B, N, D)

        # Multi-head reshape
        q = q.view(B, P, H, d).transpose(1, 2)  # (B, H, P, d)
        k = k.view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        v = v.view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        out = F.scaled_dot_product_attention(q, k, v)   # (B, H, P, d)
        out = out.transpose(1, 2).reshape(B, P, D)       # (B, P, D)
        out = self.norm(self.attn_out(out))               # (B, P, D)
        return out

    def forward(self, x):
        """
        x: (B, N, D) — full latent sequence from the encoder.
        Returns {'predicted_parameters': {panel_name -> {param_name -> (B, out_dim)}},
                 'panel_features': (B, P, D)}.
        """
        panel_feats = self._compute_panel_features(x)  # (B, P, D)

        outputs = {}

        # Physical panels: each uses its own panel feature
        for panel_name, panel_heads in self.heads.items():
            if panel_name == 'meta':
                continue
            feat = panel_feats[:, self.panel_to_idx[panel_name], :]  # (B, D)
            outputs[panel_name] = {
                pname: head(feat) for pname, head in panel_heads.items()
            }

        # Meta panel: meta.X uses panel X's feature so existence is
        # predicted from the same geometric evidence as panel X's shape.
        if 'meta' in self.heads:
            outputs['meta'] = {}
            for param_name, head in self.heads['meta'].items():
                if param_name in self.panel_to_idx:
                    feat = panel_feats[:, self.panel_to_idx[param_name], :]
                else:
                    # Rare fallback: panel not in panel_data → use mean
                    feat = panel_feats.mean(dim=1)
                outputs['meta'][param_name] = head(feat)

        return {
            'predicted_parameters': outputs,
            'panel_features': panel_feats  # For panel-level RDM loss
        }

