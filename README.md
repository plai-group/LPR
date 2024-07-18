# Layerwise Proximal Replay Code

Code for the paper **Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning**, *Jason Yoo, Yunpeng Liu, Frank Wood, Geoff Pleiss*, ICML 2024 [arxiv](https://arxiv.org/abs/2402.09542).

This repository is based on the [OCL Survey](https://github.com/AlbinSou/ocl_survey) repository.


## Installation

Clone this repository

```
git clone git@github.com:plai-group/LPR.git
```

Create a new environment with python 3.10

```
conda create -n lpr python=3.10
conda activate lpr
```

Install specific ocl_survey repo dependencies

```
pip install -r requirements.txt
```

Set your PYTHONPATH as the root of the project

```
conda env config vars set PYTHONPATH=$(pwd -P)
```

In order to let the scripts know where to fetch and log data, you should also create a **deploy config**, indicating where the results should be stored and the datasets fetched. Either add a new one or **change the content of config/deploy/default.yaml**.


## Getting Started with LPR

LPR can be enabled by setting the flag `strategy.lpr=true` in the run command. Most of LPR's implementation is located at `src/strategies/lpr_plugin.py`.

Sample Run Commands

**Split-CIFAR100**
```
cd experiments

python main.py strategy=er experiment=split_cifar100 optimizer.lr=0.1 strategy.lpr=true strategy.lpr_kwargs.preconditioner.omega_0=4 strategy.lpr_kwargs.preconditioner.beta=2 strategy.lpr_kwargs.update.every_iter=30
```

**Split-TinyImageNet**
```
cd experiments

python main.py strategy=er experiment=split_tinyimagenet optimizer.lr=0.01 strategy.lpr=true strategy.lpr_kwargs.preconditioner.omega_0=0.25 strategy.lpr_kwargs.preconditioner.beta=2 strategy.lpr_kwargs.update.every_iter=90
```

**Online CLEAR**

Download the dataset from this [link](https://drive.google.com/file/d/1wglC53ff2qGOuz6BnW6n9v2G01l5L4on/view?usp=sharing) and unzip it at `./data`.

Then, run
```
cd experiments

python main.py strategy=er experiment=clear optimizer.lr=0.01 strategy.lpr=true strategy.lpr_kwargs.preconditioner.omega_0=1. strategy.lpr_kwargs.preconditioner.beta=1. strategy.lpr_kwargs.update.every_iter=100
```


## Structure

The code is structured as follows:

```
├── avalanche.git # Avalanche-Lib code
├── config # Hydra config files
│   ├── benchmark
│   ├── best_configs # Best configs found by main_hp_tuning.py are stored here
│   ├── deploy # Contains machine specific results and data path
│   ├── evaluation # Manage evaluation frequency and parrallelism
│   ├── experiment # Manage general experiment settings
│   ├── model
│   ├── optimizer
│   ├── scheduler
│   └── strategy
├── experiments
│   ├── main_hp_tuning.py # Main script used for hyperparameter optimization
│   ├── main.py # Main script used to launch single experiments
│   └── spaces.py
├── notebooks
├── results # Exemple results structure containing results for ER
├── scripts
    └── get_results.py # Easily collect results from multiple seeds
├── src
│   ├── factories # Contains the Benchmark, Method, and Model creation
│   ├── strategies # Contains code for additional strategies or plugins
│   └── toolkit
└── tests
```

LPR specific hyperparameters can be found at `config/strategy/method_defaults.yaml`.

## Experiments launching

To launch an experiment, start from the default config file and change the part that needs to change

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel
```

It's also possible to override more fine-grained arguments

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel strategy.alpha=0.7 optimizer.lr=0.05
```

Before running the script, you can display the full config with "-c job" option

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel -c job
```

Results will be saved in the directory specified in results.yaml. Under the following structure:

```
<results_dir>/<strategy_name>_<benchmark_name>/<seed>/
```

## Citation

If you use this repo for a research project please use the following citation:

```
@misc{yoo2024layerwise,
      title={Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning}, 
      author={Jason Yoo and Yunpeng Liu and Frank Wood and Geoff Pleiss},
      year={2024},
      eprint={2402.09542},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.09542}, 
}
```
