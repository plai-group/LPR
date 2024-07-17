#!/usr/bin/env python3
import random
from types import SimpleNamespace

import ast
from functools import reduce
import numpy as np
import omegaconf
import os
import sys
import torch
from typing import Union
import wandb

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.logging import WandBLogger


def init_wandb_logger(config):
    assert config.wandb.wandb_mode in ["online", "offline", "disabled"]
    os.environ["WANDB_MODE"] = config.wandb.wandb_mode
    wandb_config = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb_config["settings"] = wandb.Settings(start_method="thread")
    return WandBLogger(project_name=config.wandb.project_name, run_name=config.wandb.run_name,
                       dir=config.wandb.wandb_root, config=wandb_config)


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def get_last_layer_name(model):
    last_layer_name = list(model.named_parameters())[-1][0].split(".")[0]

    last_layer = getattr(model, last_layer_name)

    if isinstance(last_layer, IncrementalClassifier):
        in_features = last_layer.classifier.in_features
    else:
        in_features = last_layer.in_features
    return last_layer_name, in_features


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


def extract_kwargs(extract, kwargs):
    """
    checks and extracts
    the arguments
    listed in extract
    """
    init_dict = {}
    for word in extract:
        if word not in kwargs:
            raise AttributeError(f"Missing attribute {word} in provided configuration")
        init_dict.update({word: kwargs[word]})
    return init_dict


def map_args(kwargs, keys):
    """
    Maps keys1 to keys2 in kwargs
    """
    for k1, k2 in keys.items():
        assert k1 in kwargs
        value = kwargs.pop(k1)
        kwargs[k2] = value


def clear_tensorboard_files(directory):
    for root, name, files in os.walk(directory):
        for f in files:
            if "events" in f:
                os.remove(os.path.join(root, f))


def assert_in(args, list):
    for arg in args:
        assert arg in list


def getKeys(val, old=""):
    result = []
    if isinstance(val, dict):
        if val:
            for k in val.keys():
                result.extend(getKeys(val[k], old + "." + str(k)))
    elif isinstance(val, list):
        if val:
            for i, k in enumerate(val):
                result.extend(getKeys(k, old + "." + str(i)))
    else:
        result.append(f"{old}={val}")
    return result


def preprocess_argv():
    argv = sys.argv
    new_argv = []
    for arg in argv:
        if "={" in arg:
            # nested parameter in wandb sweep
            name, content = arg.split("=")
            args = getKeys(ast.literal_eval(content), name)
            new_argv.extend(args)
        else:
            new_argv.append(arg)
    sys.argv = new_argv


def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
