################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-17-2022                                                             #
# Author: Zhiqiu Lin, Jia Shi                                                  #
# E-mail: zl279@cornell.edu, jiashi@andrew.cmu.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

""" This module contains the high-level CLEAR benchmark/factor generator.
In the original CLEAR benchmark paper (https://arxiv.org/abs/2201.06289),
a novel Streaming evaluation protocol is proposed in contrast to traditional
IID evaluation protocol for CL. The major difference lies in that:

IID Protocol: Sample a test set from current task, which requires splitting
    the data into 7:3 train:test set.
Streaming Protocol: Use the data of next task as the test set for current task,
    which is arguably more realistic since real-world model training and
    deployment usually takes considerable amount of time. By the time the
    model is applied, the task has already drifted.

We support both evaluation protocols for benchmark construction."""

from pathlib import Path
from typing import List, Sequence, Union, Any, Optional
from typing_extensions import Literal

from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_generic_benchmark_from_paths,
    create_generic_benchmark_from_tensor_lists,
)


from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List
import json
import os

import torch
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.utils import default_flist_reader
from avalanche.benchmarks.datasets.clear import clear_data

EVALUATION_PROTOCOLS = ["iid", "streaming"]

_CLEAR_DATA_SPLITS = {"clear10", "clear100", "clear10_neurips2021", "clear100_cvpr2022"}

CLEAR_FEATURE_TYPES = {
    "clear10": ["moco_b0"],
    "clear100": ["moco_b0"],
    "clear10_neurips2021": ["moco_b0", "moco_imagenet", "byol_imagenet", "imagenet"],
    "clear100_cvpr2022": ["moco_b0"],
}

SPLIT_OPTIONS = ["all", "train", "test"]

SEED_LIST = [0, 1, 2, 3, 4]  # Available seeds for train:test split


def _load_json(json_location):
    with open(json_location, "r") as f:
        obj = json.load(f)
    return obj


class CLEARDataset(DownloadableDataset):
    """CLEAR Base Dataset for downloading / loading metadata"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = False,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This base class simply download and unzip the CLEAR dataset.

        This serves as a base class for _CLEARImage/_CLEARFeature dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location(data_name)

        assert data_name in _CLEAR_DATA_SPLITS
        self.data_name = data_name
        self.module = clear_data
        self._paths_and_targets: List[List[Tuple[str, int]]] = []

        super(CLEARDataset, self).__init__(root, download=download, verbose=True)
        self._load_dataset()

    def _download_dataset(self) -> None:
        target_module = getattr(self.module, self.data_name)

        for name, base_url in target_module:
            if self.verbose:
                print("Downloading " + name + "...")
            url = os.path.join(base_url, name)
            self._download_and_extract_archive(
                url=url, file_name=name, checksum=None, remove_archive=True
            )

    def _load_metadata(self) -> bool:
        if "_" in self.data_name:
            return self._load_metadata_old()
        else:
            return self._load_metadata_new()

    def _load_metadata_old(self) -> bool:
        raise NotImplementedError()

    def _load_metadata_new(self) -> bool:
        splits = ["train", "test"] if self.split == "all" else [self.split]
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False

            self.labeled_metadata = _load_json(
                train_folder_path / "labeled_metadata.json"
            )

            class_names_file = train_folder_path / "class_names.txt"
            self.class_names = class_names_file.read_text().split("\n")

            self.samples = []
            self._paths_and_targets = []
            for bucket, data in self.labeled_metadata.items():
                for class_idx, class_name in enumerate(self.class_names):
                    metadata_path = data[class_name]
                    metadata_path = train_folder_path / metadata_path
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist. ")
                        return False
                    metadata = _load_json(metadata_path)
                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        self.samples.append((f_path, class_idx))

            # Check whether all labeled images exist
            for img_path, _ in self.samples:
                path = self.root / img_path
                if not os.path.exists(path):
                    print(f"{path} does not exist.")
                    return False
        return True

    def _download_error_message(self) -> str:
        all_urls = [
            os.path.join(item[1], item[0])
            for item in getattr(self.module, self.data_name)
        ]

        base_msg = (
            f"[{self.data_name}] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        return str(self.root / img_path), target

    def __len__(self):
        return len(self.samples)


class _CLEARImage(CLEARDataset):
    """CLEAR Image Dataset (base class for CLEARImage)"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = True,
        split: str = "all",
        seed: Optional[int] = None,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This image dataset will contain samples from all buckets of CLEAR,
        so it is not intended for CL purposes. It simply download and
        unzip the CLEAR dataset.

        Paths and targets for each bucket for benchmark creation will be
        loaded in self._paths_and_targets ;
        can use self.get_paths_and_targets() with root appended to each path


        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        :param split: Choose from ['all', 'train', 'test'].
            If 'all', then return all data from all buckets.
            If 'train'/'test', then only return train/test data.
        :param seed: The random seed used for splitting the train:test into 7:3
            If split=='all', then seed must be None (since no split is done)
            otherwise, choose from [0, 1, 2, 3, 4]
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        """
        self.split = split
        assert self.split in SPLIT_OPTIONS, "Invalid split option"
        if self.split == "all":
            assert seed is None, "Specify a seed if not splitting train:test"
        else:
            assert seed in SEED_LIST
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.paths: List[Union[str, Path]] = []

        self.class_names: List[str] = []
        """
        After _load_metadata(), the class names will be loaded in order
        aligned with target index.
        """

        super(_CLEARImage, self).__init__(
            root, data_name=data_name, download=download, verbose=True
        )

    def _load_metadata(self) -> bool:
        if "_" in self.data_name:
            return self._load_metadata_old()
        else:
            return self._load_metadata_new()

    def _load_metadata_old(self) -> bool:
        raise NotImplementedError()

    def _load_metadata_new(self) -> bool:
        if not super(_CLEARImage, self)._load_metadata_new():
            print("CLEAR has not yet been downloaded")
            return False

        self.paths = []
        self.targets = []
        self._paths_and_targets = []
        splits = ["test", "train"] if self.split == "all" else [self.split]
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False

            self.labeled_metadata = _load_json(
                train_folder_path / "labeled_metadata.json"
            )

            for bucket, data in self.labeled_metadata.items():
                samples = []
                for class_idx, class_name in enumerate(self.class_names):
                    metadata_path = data[class_name]
                    metadata_path = train_folder_path / metadata_path
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist. ")
                        return False
                    metadata = _load_json(metadata_path)
                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        samples.append((f_path, class_idx))
                if self.split == "all" and split == "train":
                    _samples = self._paths_and_targets[int(bucket)]
                    _samples += samples
                    self._paths_and_targets[int(bucket)] = _samples
                else:
                    self._paths_and_targets.append(samples)
        for path_and_target_list in self._paths_and_targets:
            for img_path, target in path_and_target_list:
                self.paths.append(self.root / img_path)
                self.targets.append(target)
        return True

    def get_paths_and_targets(
        self, root_appended=True
    ) -> Sequence[Sequence[Tuple[Union[str, Path], int]]]:
        """Return self._paths_and_targets with root appended or not"""
        if not root_appended:
            return self._paths_and_targets
        else:
            paths_and_targets: List[List[Tuple[Path, int]]] = []
            for path_and_target_list in self._paths_and_targets:
                paths_and_targets.append([])
                for img_path, target in path_and_target_list:
                    paths_and_targets[-1].append((self.root / img_path, target))
            return paths_and_targets

    def __getitem__(self, index):
        img_path = self.paths[index]
        target = self.targets[index]

        img = self.loader(str(self.root / img_path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


class _CLEARFeature(CLEARDataset):
    """CLEAR Feature Dataset (base class for CLEARFeature)"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = True,
        split: str = "all",
        seed: Optional[int] = None,
        feature_type: str = "moco_b0",
        target_transform=None,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This image dataset will contain samples from all buckets of CLEAR,
        so it is not intended for CL purposes. It simply download and
        unzip the CLEAR dataset.

        Tensors and targets for benchmark creation will be
        loaded in self.tensors_and_targets

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        :param split: Choose from ['all', 'train', 'test'].
            If 'all', then return all data from all buckets.
            If 'train'/'test', then only return train/test data.
        :param seed: The random seed used for splitting the train:test into 7:3
            If split=='all', then seed must be None (since no split is done)
            otherwise, choose from [0, 1, 2, 3, 4]
        :param feature_type: The type of features.
            For CLEAR10_NeurIPS2021, choose from [
                'moco_b0', # Moco V2 ResNet50 pretrained on bucket 0
                'moco_imagenet', # Moco V2 ResNet50 pretrained on Imagenet
                'byol_imagenet', # BYOL ResNet50 pretrained on Imagenet
                'imagenet', # ResNet50 pretrained on Imagenet
            ]
            For other datasets: 'moco_b0' only
        :param target_transform: The transformations to apply to the Y values.
        """
        self.split = split
        assert self.split in ["all", "train", "test"], "Invalid split option"

        if self.split == "all":
            assert seed is None, "Specify a seed if not splitting train:test"
        else:
            assert seed in SEED_LIST
        self.seed = seed

        self.feature_type = feature_type
        assert feature_type in CLEAR_FEATURE_TYPES[data_name]
        self.target_transform = target_transform

        self.tensors_and_targets: List[Tuple[List[torch.Tensor], List[int]]] = []

        super(_CLEARFeature, self).__init__(
            root, data_name=data_name, download=download, verbose=True
        )

    def _load_metadata(self) -> bool:
        if "_" in self.data_name:
            return self._load_metadata_old()
        else:
            return self._load_metadata_new()

    def _load_metadata_old(self) -> bool:
        raise NotImplementedError()

    def _load_metadata_new(self) -> bool:
        if not super(_CLEARFeature, self)._load_metadata_new():
            print("CLEAR has not yet been downloaded")
            return False

        self.tensors_and_targets = []
        splits = ["test", "train"] if self.split == "all" else [self.split]
        for split in splits:
            folder_path = self.root / self.split
            feature_folder_path = folder_path / "features" / self.feature_type
            metadata = _load_json(feature_folder_path / "features.json")
            tensors = []
            targets = []
            for bucket, data in metadata.items():
                for class_idx, class_name in enumerate(self.class_names):
                    feature_path = data[class_name]
                    try:
                        features = torch.load(folder_path / feature_path)
                    except Exception as e:
                        print(f"Error loading {feature_path}")
                        return False
                    for _id, tensor in features.items():
                        tensors.append(tensor)
                        targets.append(class_idx)
                if self.split == "all" and split == "train":
                    _tensors, _targets = self.tensors_and_targets[int(bucket)]
                    _tensors += tensors
                    _targets += targets
                    self.tensors_and_targets[int(bucket)] = (_tensors, _targets)
                else:
                    self.tensors_and_targets.append((tensors, targets))

        self.tensors = []
        self.targets = []
        for tensors, targets in self.tensors_and_targets:
            for tensor, target in zip(tensors, targets):
                self.tensors.append(tensor)
                self.targets.append(target)

        return True

    def __getitem__(self, index):
        tensor = self.tensors[index]
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return tensor, target

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":
    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    data_names = ["clear10_neurips2021", "clear100_cvpr2022", "clear10", "clear100"]
    for data_name in data_names:
        root = f"../avalanche_datasets/{data_name}"
        print(root)
        if not os.path.exists(root):
            Path(root).mkdir(parents=True)
        clear_dataset_all = _CLEARImage(
            root=root,
            data_name=data_name,
            download=True,
            split="all",
            seed=None,
            transform=transform,
        )
        clear_dataset_train = _CLEARImage(
            root=root,
            data_name=data_name,
            download=True,
            split="train",
            seed=0,
            transform=transform,
        )
        clear_dataset_test = _CLEARImage(
            root=root,
            data_name=data_name,
            download=True,
            split="test",
            seed=0,
            transform=transform,
        )
        print(f"{data_name} size (all): ", len(clear_dataset_all))
        print(f"{data_name} size (train): ", len(clear_dataset_train))
        print(f"{data_name} size (test): ", len(clear_dataset_test))

        clear_dataset_train_feature = _CLEARFeature(
            root=root,
            data_name=data_name,
            download=True,
            feature_type="moco_b0",
            split="train",
            seed=0,
        )
        print("clear10 size (train features): ", len(clear_dataset_train_feature))
        if "_" in data_name:
            clear_dataset_all_feature = _CLEARFeature(
                root=root,
                data_name=data_name,
                download=True,
                feature_type="moco_b0",
                split="all",
                seed=None,
            )
            clear_dataset_test_feature = _CLEARFeature(
                root=root,
                data_name=data_name,
                download=True,
                feature_type="moco_b0",
                split="test",
                seed=0,
            )
            print(
                f"{data_name} size (test features): ", len(clear_dataset_test_feature)
            )
            print(f"{data_name} size (all features): ", len(clear_dataset_all_feature))
        print("Classes are: ")
        for i, name in enumerate(clear_dataset_test.class_names):
            print(f"{i} : {name}")
        dataloader = DataLoader(clear_dataset_test_feature, batch_size=1)

        for batch_data in dataloader:
            x, y = batch_data
            print(x.size())
            print(len(y))
            break


def CLEAR(
    *,
    data_name: str = "clear10",
    evaluation_protocol: str = "iid",
    feature_type: Optional[str] = None,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None,
):
    """
    Creates a Domain-Incremental benchmark for CLEAR 10 & 100
    with 10 & 100 illustrative classes and an n+1 th background class.

    If the dataset is not present in the computer, **this method will be
    able to automatically download** and store it.

    This generator supports benchmark construction of both 'iid' and 'streaming'
    evaluation_protocol. The main difference is:

    'iid': Always sample testset from current task, which requires
        splitting the data into 7:3 train:test with a given random seed.
    'streaming': Use all data of next task as the testset for current task,
        which does not split the data and does not require random seed.


    The generator supports both Image and Feature (Tensor) datasets.
    If feature_type == None, then images will be used.
    If feature_type is specified, then feature tensors will be used.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    Note that the train/test streams will still be data of current task,
    regardless of whether evaluation protocol is 'iid' or 'streaming'.
    For 'iid' protocol, train stream is 70% of current task data,
    and test stream is 30% of current task data.
    For 'streaming' protocol, train stream is 100% of current task data,
    and test stream is just a duplicate of train stream.

    The task label 0 will be assigned to each experience.

    :param evaluation_protocol: Choose from ['iid', 'streaming']
        if chosen 'iid', then must specify a seed between [0,1,2,3,4];
        if chosen 'streaming', then the seed will be ignored.
    :param feature_type: Whether to return raw RGB images or feature tensors
        extracted by pre-trained models. Can choose between
        [None, 'moco_b0', 'moco_imagenet', 'byol_imagenet', 'imagenet'].
        If feature_type is None, then images will be returned.
        Otherwise feature tensors will be returned.
    :param seed: If evaluation_protocol is iid, then must specify a seed value
        for train:test split. Choose between [0,1,2,3,4].
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        str(data_name) will be used.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """
    assert data_name in _CLEAR_DATA_SPLITS

    assert evaluation_protocol in EVALUATION_PROTOCOLS, (
        "Must specify a evaluation protocol from " f"{EVALUATION_PROTOCOLS}"
    )

    if evaluation_protocol == "streaming":
        assert seed is None, (
            "Seed for train/test split is not required " "under streaming protocol"
        )
        train_split = "all"
        test_split = "all"
    elif evaluation_protocol == "iid":
        assert seed in SEED_LIST, "No seed for train/test split"
        train_split = "train"
        test_split = "test"
    else:
        raise NotImplementedError()

    if dataset_root is not None:
        dataset_root = os.path.join(dataset_root, "clear")

    if feature_type is None:
        clear_dataset_train = _CLEARImage(
            root=dataset_root,
            data_name=data_name,
            download=True,
            split=train_split,
            seed=seed,
            transform=train_transform,
        )
        clear_dataset_test = _CLEARImage(
            root=dataset_root,
            data_name=data_name,
            download=True,
            split=test_split,
            seed=seed,
            transform=eval_transform,
        )
        train_samples_paths = clear_dataset_train.get_paths_and_targets(
            root_appended=True
        )
        test_samples_paths = clear_dataset_test.get_paths_and_targets(
            root_appended=True
        )
        benchmark_obj = create_generic_benchmark_from_paths(
            train_samples_paths,
            test_samples_paths,
            task_labels=[0 for _ in range(len(train_samples_paths))],
            complete_test_set_only=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        clear_dataset_train = _CLEARFeature(
            root=dataset_root,
            data_name=data_name,
            download=True,
            feature_type=feature_type,
            split=train_split,
            seed=seed,
        )
        clear_dataset_test = _CLEARFeature(
            root=dataset_root,
            data_name=data_name,
            download=True,
            feature_type=feature_type,
            split=test_split,
            seed=seed,
        )
        train_samples = clear_dataset_train.tensors_and_targets
        test_samples = clear_dataset_test.tensors_and_targets

        benchmark_obj = create_generic_benchmark_from_tensor_lists(
            train_samples,
            test_samples,
            task_labels=[0 for _ in range(len(train_samples))],
            complete_test_set_only=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    return benchmark_obj


class CLEARMetric:
    """All metrics used in CLEAR paper.
    More information can be found at:
    https://clear-benchmark.github.io/
    """

    def __init__(self):
        super(CLEARMetric, self).__init__()

    def get_metrics(self, matrix):
        """Given an accuracy matrix, returns the 5 metrics used in CLEAR paper

        These are:
            'in_domain' : In-domain accuracy (avg of diagonal)
            'next_domain' : In-domain accuracy (avg of superdiagonal)
            'accuracy' : Accuracy (avg of diagonal + lower triangular)
            'backward_transfer' : BwT (avg of lower triangular)
            'forward_transfer' : FwT (avg of upper triangular)

        :param matrix: Accuracy matrix,
            e.g., matrix[5][0] is the test accuracy on 0-th-task at timestamp 5
        :return: A dictionary containing these 5 metrics
        """
        assert matrix.shape[0] == matrix.shape[1]
        metrics_dict = {
            "in_domain": self.in_domain(matrix),
            "next_domain": self.next_domain(matrix),
            "accuracy": self.accuracy(matrix),
            "forward_transfer": self.forward_transfer(matrix),
            "backward_transfer": self.backward_transfer(matrix),
        }
        return metrics_dict

    def accuracy(self, matrix):
        """
        Average of lower triangle + diagonal
        Evaluate accuracy on seen tasks
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i + 1)]
        return sum(res) / len(res)

    def in_domain(self, matrix):
        """
        Diagonal average
        Evaluate accuracy on the current task only
        """
        r, _ = matrix.shape
        res = [matrix[i, i] for i in range(r)]
        return sum(res) / r

    def next_domain(self, matrix):
        """
        Superdiagonal average
        Evaluate on the immediate next timestamp
        """
        r, _ = matrix.shape
        res = [matrix[i, i + 1] for i in range(r - 1)]
        return sum(res) / (r - 1)

    def forward_transfer(self, matrix):
        """
        Upper trianglar average
        Evaluate generalization to all future task
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i + 1, r)]
        return sum(res) / len(res)

    def backward_transfer(self, matrix):
        """
        Lower triangular average
        Evaluate learning without forgetting
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i)]
        return sum(res) / len(res)


__all__ = ["CLEAR", "CLEARMetric"]

if __name__ == "__main__":
    import sys
    from torchvision import transforms
    import torch

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_name = "clear10"
    root = f"../avalanche_datasets/{data_name}"

    for p in EVALUATION_PROTOCOLS:
        seed_list: Sequence[Optional[int]]
        if p == "streaming":
            seed_list = [None]
        else:
            seed_list = SEED_LIST

        for f in [None] + CLEAR_FEATURE_TYPES[data_name]:
            t = transform if f is None else None
            for seed in seed_list:
                benchmark_instance = CLEAR(
                    evaluation_protocol=p,
                    feature_type=f,
                    seed=seed,
                    train_transform=t,
                    eval_transform=t,
                    dataset_root=root,
                )
                benchmark_instance.train_stream[0]
                # check_vision_benchmark(benchmark_instance)
                print(
                    f"Check pass for {p} protocol, and "
                    f"feature type {f} and seed {seed}"
                )
    sys.exit(0)
