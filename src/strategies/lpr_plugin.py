from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
import wandb


@dataclass
class PreconditionerInv:
    matrix: torch.Tensor
    has_bias: bool
    parallel: bool = False


class LPRPlugin(SupervisedPlugin):
    def __init__(self, lpr_kwargs, storage_policy):
        # NOTE: Refer to config/strategy/method_defaults.yaml for hyperparameter information
        self.preconditioner_config = lpr_kwargs['preconditioner']
        self.update_config = lpr_kwargs['update']
        self.log_config = lpr_kwargs['log']
        self.storage_policy = storage_policy

        # Computed preconditioner inverses are stored here until they are updated
        self._preconditioner_invs: Dict[str, PreconditionerInv] = dict()

    @torch.no_grad()
    def before_backward(self, strategy, **kwargs):
        """
        Check if layerwise preconditioners should be updated.
        """
        if self.storage_policy is None:
            assert hasattr(strategy, "storage_policy") and strategy.storage_policy is not None
            self.storage_policy = strategy.storage_policy
        if self.update_config.every_iter is None:
            self.update_config.every_iter = strategy.train_epochs
        if self._should_update_preconditioners(strategy):
            self._set_preconditioner_invs(strategy)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Precondition gradients in a layerwise manner.
        """
        if len(self.storage_policy.buffer) == 0:
            return

        log_grad_norms = self.log_config.grad_norms\
            and strategy.clock.train_exp_counter % self.log_config.grad_norms == 0\
            and strategy.clock.train_exp_iterations in [0, strategy.train_epochs-1]

        precondition_grad_args = dict(model=strategy.model, calculate_norm=log_grad_norms,
                                      preconditioner_invs=self._preconditioner_invs)
        result_dict, grad_norm_dict = precondition_gradient(**precondition_grad_args)
        if log_grad_norms:
            log_grad_norms_to_wandb(result_dict, grad_norm_dict)

    def _set_preconditioner_invs(self, strategy):
        """
        Set preconditioning matrices for subsequent experiences
        """
        layer_stats = self._compute_layer_statistics(strategy)
        self._preconditioner_invs = self._compute_preconditioner_invs(layer_stats=layer_stats)

    def _should_update_preconditioners(self, strategy) -> bool:
        return len(self.storage_policy.buffer) > 0 \
                and strategy.clock.train_iterations % self.update_config.every_iter == 0

    @torch.no_grad()
    def _compute_layer_statistics(self, strategy):
        """
        Compute the scaled uncentered covariance matrix omega^l Z^l'Z^l for all layers.
        """
        result = None
        model, device = strategy.model, strategy.device

        def retain_activations(layers_info, layer_name):
            def hook(model, input, output):
                layers_info[layer_name] = dict(module=model, acts=input[0].detach())
            return hook

        n_seen = 0
        n_data = len(self.storage_policy.buffer) if self.update_config.n_data is None else self.update_config.n_data
        replay_loader = torch.utils.data.DataLoader(self.storage_policy.buffer, shuffle=True,
                                                    batch_size=min(self.update_config.batch_size, n_data))

        for batch in replay_loader:
            if n_seen >= n_data:
                break
            X_batch = batch[0].to(device)
            n_seen += len(X_batch)

            # setup forward hooks
            layers_info, hooks = dict(), []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
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
                result = {k: dict(ucov=0., n_samples=0, has_bias=False) for k in layers_info.keys()}

            # Update result with layerwise Fisher information calculated from this batch for all layers.
            for layer_name in result.keys():
                layer_info = layers_info[layer_name]
                module, acts = layer_info['module'], layer_info['acts']

                # Compute this batch's uncentered covariance and other miscellaneous statistics.
                # scaling for conv and bn is 1 and n_acts_per_data when modifier is 0 an 1 respectively.
                if isinstance(module, nn.Conv2d):
                    ucov, n_acts_per_data, has_bias = compute_conv_ucov(acts, module)
                elif isinstance(module, nn.Linear):
                    ucov, n_acts_per_data, has_bias = compute_linear_ucov(acts, module)
                elif isinstance(module, nn.BatchNorm2d):
                    ucovs, n_acts_per_data, has_bias = compute_bn_ucov(acts, module)
                else:
                    raise NotImplementedError()
                omega_0 = get_omega_0(self.preconditioner_config, module)
                c_layer = get_c_layer(self.preconditioner_config, module, n_acts_per_data)

                # Update result with layerwise Fisher information calculated from this batch for this layer.
                omega_0_layer = omega_0 / c_layer
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    result[layer_name]["ucov"] += omega_0_layer * ucov.to(device)
                else:
                    if isinstance(result[layer_name]["ucov"], float):
                        result[layer_name]["ucov"] = [0.] * len(ucovs)
                    for c, channel_P in enumerate(ucovs):
                        result[layer_name]["ucov"][c] += omega_0_layer * channel_P.to(device)
                result[layer_name]["n_samples"] += len(X_batch)
                result[layer_name]["has_bias"] = has_bias

        for stats in result.values():
            if isinstance(stats["ucov"], list):
                stats["ucov"] = [prec / stats["n_samples"] for prec in stats["ucov"]]
            else:
                stats["ucov"] = stats["ucov"] / stats["n_samples"]
        return result

    @torch.no_grad()
    def _compute_preconditioner_invs(self, layer_stats: Dict[str, Any]):
        """
        Add identity matrix to the scaled uncentered covariance matrix and invert for all layers.
        """

        def ucov_to_preconditioner_inv(ucov, has_bias):
            identity = torch.eye(ucov.size(0)).to(ucov.device)
            preconditioner_inv = (ucov + identity).inverse()
            return PreconditionerInv(matrix=preconditioner_inv, has_bias=has_bias, parallel=False)

        preconditioner_invs = dict()
        for layer_name, stats in layer_stats.items():
            layer_ucov, has_bias = stats["ucov"], stats["has_bias"]
            if isinstance(layer_ucov, list):
                # Currently happens only for BatchNorm2D, where each channel parameter receives different inputs.
                layer_prec_invs = [ucov_to_preconditioner_inv(ucov=ucov, has_bias=has_bias) for ucov in layer_ucov]
                stacked_matrix = torch.stack([layer_prec_inv.matrix for layer_prec_inv in layer_prec_invs], dim=0)
                preconditioner_invs[layer_name] = PreconditionerInv(stacked_matrix, has_bias, True)
            else:
                preconditioner_invs[layer_name] = ucov_to_preconditioner_inv(ucov=layer_ucov, has_bias=has_bias)
        return preconditioner_invs


def compute_conv_ucov(acts, module):
    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    padding_mode = module.padding_mode
    dilation = module.dilation
    has_bias = module.bias is not None
    assert padding_mode == "zeros"

    acts_unfolded = nn.Unfold(kernel_size, dilation, padding, stride)(acts)
    acts_unfolded = acts_unfolded.transpose(-1, -2).reshape(-1, acts_unfolded.size(1))
    if has_bias:
        ones = torch.ones(acts_unfolded.size(0), 1).to(acts.device)
        acts_unfolded = torch.cat((acts_unfolded, ones), dim=-1)
    ucov = acts_unfolded.T @ acts_unfolded
    n_acts_per_data = int(acts_unfolded.size(0) / acts.size(0))
    return ucov, n_acts_per_data, has_bias


def compute_linear_ucov(acts, module):
    has_bias = module.bias is not None
    if has_bias:
        ones = torch.ones(acts.size(0), 1).to(acts.device)
        acts = torch.cat((acts, ones), dim=-1)
    ucov = acts.T @ acts
    n_acts_per_data = 1
    return ucov, n_acts_per_data, has_bias


def compute_bn_ucov(acts, module):
    has_bias = module.bias is not None
    acts_flattened = acts.transpose(0, 1).flatten(1)  # num_channels x (n_batch * width * height)
    if module.track_running_stats:
        mean = module.running_mean
        var = module.running_var
    else:
        mean = acts_flattened.mean()
        var = acts_flattened.var(unbiased=False)

    acts_flattened = (acts_flattened - mean.view(-1, 1)) / (var.view(-1, 1) + module.eps)
    channel_ucovs = []
    for c in range(acts_flattened.size(0)):
        channel_acts = acts_flattened[c]
        if has_bias:
            ones = torch.ones(channel_acts.size(0)).to(acts.device)
            channel_acts = torch.stack((channel_acts, ones), dim=0)
        channel_ucov = channel_acts @ channel_acts.T
        channel_ucovs.append(channel_ucov)

    n_acts_per_data = acts.size(-2) * acts.size(-1)  # width x height
    return channel_ucovs, n_acts_per_data, has_bias


def precondition_gradient(model: nn.Module, preconditioner_invs: Dict[str, torch.Tensor],
                          calculate_norm: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    result_dict, layer_grad_norm_dict = dict(), dict()
    total_g_norm, total_g_new_norm = 0., 0.

    # Project each layer's gradient
    for module_name, module in model.named_modules():
        if module_name not in preconditioner_invs:
            continue

        module_info = preconditioner_invs[module_name]
        precondition_args = dict(preconditioner_inv=module_info.matrix, weight=module.weight, bias=module.bias,
                                 parallel=module_info.parallel, calculate_norm=calculate_norm)
        preconditioning_stats = precondition_layer_gradient(**precondition_args)

        if calculate_norm:
            g_norm, g_new_norm = preconditioning_stats["g_norm"], preconditioning_stats["g_new_norm"]
            total_g_norm, total_g_new_norm = total_g_norm + g_norm ** 2, total_g_new_norm + g_new_norm ** 2
            layer_grad_norm_dict[module_name] = dict(g=g_norm, g_new=g_new_norm, g_norm_ratio=g_new_norm/g_norm)

    if calculate_norm and len(preconditioner_invs) > 0:
        result_dict = {"Grad Norm": total_g_norm ** 0.5, "Projected Grad Norm": total_g_new_norm ** 0.5,
                       "Grad Norm Projection Ratio": total_g_new_norm ** 0.5 / total_g_norm ** 0.5}
    return result_dict, layer_grad_norm_dict


def precondition_layer_gradient(preconditioner_inv: torch.Tensor, weight: nn.Parameter, bias: nn.Parameter,
                                parallel: bool = False, calculate_norm: bool = False) -> Dict[str, int]:
    has_bias = bias is not None

    if has_bias:
        g_weight = weight.grad.data.view(weight.size(0), -1)
        g_bias = bias.grad.data.view(bias.size(0), -1)
        g = torch.cat((g_weight, g_bias), dim=-1)
    else:
        g = weight.grad.data.view(weight.size(0), -1)

    expr = 'CD,CDE->CE' if parallel else 'CD,DE->CE'
    g_new = torch.einsum(expr, g, preconditioner_inv)

    if has_bias:
        g_weight_new = g_new[..., :-1]
        g_bias_new = g_new[..., -1:]
        weight.grad.data = g_weight_new.view(weight.size())
        bias.grad.data = g_bias_new.view(bias.size())
    else:
        weight.grad.data = g_new.view(weight.size())

    result = dict(g_norm=g.norm(2).item(), g_new_norm=g_new.norm(2).item()) if calculate_norm else dict()
    return result


def get_omega_0(preconditioner_config, module: nn.Module) -> float:
    omega_0 = preconditioner_config.omega_0
    if isinstance(module, nn.Linear):
        if preconditioner_config.linear_omega_0 is not None:
            omega_0 = preconditioner_config.linear_omega_0
    elif isinstance(module, nn.Conv2d):
        if preconditioner_config.conv_omega_0 is not None:
            omega_0 = preconditioner_config.conv_omega_0
    elif isinstance(module, nn.BatchNorm2d):
        if preconditioner_config.bn_omega_0 is not None:
            omega_0 = preconditioner_config.bn_omega_0
    else:
        raise NotImplementedError()
    return omega_0


def get_c_layer(preconditioner_config, module: nn.Module, n_acts_per_data: int = 1) -> float:
    beta = preconditioner_config.beta
    if isinstance(module, nn.Linear):
        return 1.
    elif isinstance(module, nn.Conv2d):
        if preconditioner_config.conv_beta is not None:
            beta = preconditioner_config.conv_beta
    elif isinstance(module, nn.BatchNorm2d):
        if preconditioner_config.bn_beta is not None:
            beta = preconditioner_config.bn_beta
    else:
        raise NotImplementedError()
    c_layer = n_acts_per_data ** beta
    return c_layer


def log_grad_norms_to_wandb(result_dict, grad_norm_dict):
    wandb_dict = result_dict
    for module_name, info in grad_norm_dict.items():
        wandb_dict[f"{module_name}_Grad_Norm"] = info["g"]
        wandb_dict[f"{module_name}_Proj_Grad_Norm"] = info["g_new"]
        wandb_dict[f"{module_name}_Proj_Norm_Ratio"] = info["g_norm_ratio"]
        if "g_scaled" in info:
            wandb_dict[f"{module_name}_Scaled_Norm"] = info["g_scaled"]
    wandb_dict = {f"LPR_Grad_Info/{k}": v for k, v in wandb_dict.items()}
    wandb.log(wandb_dict, commit=False)
