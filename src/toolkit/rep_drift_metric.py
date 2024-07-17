from avalanche.benchmarks.utils.data_loader import DataLoader
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name
import torch
from typing import Dict, List, Tuple

from src.toolkit.slim_resnet18 import ResNet


# FIXME: This doesn't work with EMA since that requires model to be in eval mode
# NOTE: Training data has random feature augmentations applied to them so you can't measure rep drift as of now.

class RepresentationDriftMetric(PluginMetric[Dict[str, float]]):  # Consider subclassing GenericPluginMetric
    """Every specified number of iteration, plot how much representations in different ResNet layer has
    diverged from its previous measurement on the eval and baseline task datapoints.
    """
    def __init__(self, data_dict, every_iter: int = 1, n_samples: int = 100,
                 selected_layer_names: List[str] = ['layer1', 'layer2', 'layer3', 'layer4', 'linear.classifier']):
        self._every_iter: int = every_iter
        self._selected_layer_names: List[str] = selected_layer_names
        self._n_samples = n_samples

        self._loader_dict: Dict[str, DataLoader] = dict()
        self._loader_activation_dict: Dict[str, Dict[str, torch.Tensor]] = dict()
        self._loader_displacement_dict: Dict[str, Dict[str, Tuple[float, float]]] = dict()

        for data_name, data in data_dict.items():
            data = data[0].eval()._dataset
            self._loader_dict[data_name] = DataLoader(data, batch_size=min(n_samples, len(data)))

    def result(self):
        result_dict = dict()
        for loader_name, displacement_dict in self._loader_displacement_dict.items():
            curr_dict, total_dict = _summarize_displacement(displacement_dict, f"Rep_Drift/{loader_name}")
            result_dict.update(curr_dict)
            result_dict.update(total_dict)
        return result_dict

    def reset(self):
        pass

    def after_training_iteration(self, strategy, **kwargs):
        super().after_training_iteration(strategy)
        assert isinstance(strategy.model, ResNet)
        if strategy.clock.train_iterations % self._every_iter > 0:
            return

        with torch.no_grad():
            for loader_name, loader in self._loader_dict.items():
                old_act_dict = self._loader_activation_dict.get(loader_name, dict())
                old_disp_dict = self._loader_displacement_dict.get(loader_name, dict())
                new_act_dict = _compute_activations(strategy, loader, self._n_samples, self._selected_layer_names)
                new_disp_dict = _updated_displacement(old_disp_dict, old_act_dict, new_act_dict)
                self._loader_activation_dict[loader_name] = new_act_dict
                self._loader_displacement_dict[loader_name] = new_disp_dict
            return self._package_result(strategy.clock.train_iterations)

    def _package_result(self, train_iterations) -> "MetricResult":
        metrics = []
        for k, v in self.result().items():
            # metric_name = get_metric_name(self, strategy, add_experience=False, add_task=k)
            metrics.append(MetricValue(self, k, v, train_iterations))
        return metrics


def _compute_activations(strategy, dataloader, n_samples, selected_layer_names) -> Dict[str, float]:
    result = None
    model, device = strategy.model, strategy.device

    def retain_activations(layers_info, layer_name):
        def hook(model, input, output):
            layers_info[layer_name] = output.detach()
        return hook

    n_seen = 0
    for batch in dataloader:
        X_batch = batch[0].to(device)
        # setup forward hooks
        layers_info, hooks = dict(), []
        for name, module in model.named_modules():
            if name in selected_layer_names:
                hook = module.register_forward_hook(retain_activations(layers_info, name))
                hooks.append(hook)
        # run model forward
        is_training = strategy.is_training
        strategy.is_training = False
        model.eval()
        model(X_batch)
        if is_training:
            model.train()
            strategy.is_training = True
        # remove forward hooks
        for hook in hooks:
            hook.remove()
        if result is None:
            result = layers_info
        else:
            result = {k: torch.stack((v, layers_info[k]), dim=0) for k, v in result.items()}

        n_seen += len(X_batch)
        if n_seen >= n_samples:
            break
    return result


def _updated_displacement(displacement_dict, old_activation_dict, new_activation_dict):
    new_displacement_dict = dict()
    for layer_name, new_acts in new_activation_dict.items():
        old_acts = old_activation_dict.get(layer_name, 0.)
        old_displacement = displacement_dict.get(layer_name, [0., 0.])[1]
        if isinstance(old_acts, torch.Tensor) and old_acts.shape != new_acts.shape:
            new_acts = new_acts[:, :old_acts.size(-1)]  # This line triggers when classification head size increases.
        current_displacement = ((old_acts - new_acts) ** 2).sum() ** 0.5 / new_acts.size(0)
        total_displacement = ((old_displacement**2 + current_displacement**2)**0.5)
        new_displacement_dict[layer_name] = (current_displacement.item(), total_displacement.item())
    return new_displacement_dict


def _summarize_displacement(displacement_dict: Dict[str, Tuple[float, float]], prefix: str
                            ) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Returns a dict that contains L2-norm differences of each ResNet blocks and linear layers.
    # TODO: Make this more general instead of just working with Slim ResNet18.
    result_curr, result_total = {f'{prefix}/Current/All': 0.}, {f'{prefix}/Total/All': 0.}
    for layer_name, (delta_curr, delta_total) in displacement_dict.items():
        layer_group = layer_name.split('.')[0]
        key_curr, key_total = f'{prefix}/Current/{layer_group}', f'{prefix}/Total/{layer_group}'
        result_curr[key_curr] = delta_curr
        result_total[key_total] = delta_total
        result_curr[f'{prefix}/Current/All'] = (result_curr[f'{prefix}/Current/All']**2 + delta_curr**2)**0.5
        result_total[f'{prefix}/Total/All'] = (result_total[f'{prefix}/Total/All']**2 + delta_total**2)**0.5
    return result_curr, result_total
