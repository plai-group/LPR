#!/usr/bin/env python3
import os
import sys
sys.path.append('..')  # noqa

import hydra
import omegaconf
import pandas as pd
import torch
import wandb

import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import src.toolkit.utils as utils
from avalanche.benchmarks.scenarios import OnlineCLScenario
from src.factories.benchmark_factory import DS_SIZES


@hydra.main(config_path="../config", config_name="config.yaml")
def main(config):
    wandb_logger = utils.init_wandb_logger(config)
    utils.set_seed(config.experiment.seed)

    plugins = []

    scenario = benchmark_factory.create_benchmark(
        **config["benchmark"].factory_args,
        dataset_root=config.benchmark.dataset_root,
        seed=config["experiment"].seed if config.benchmark.dataset_name == "clear" else None
    )

    # Transforms have randomness built in and we don't want that if we're doing representation comparison
    if config.evaluation.rep_drift.enable:
        data_dict = dict(First_Task_Test=scenario.test_stream[:1])
    else:
        data_dict = None

    model = model_factory.create_model(
        **config["model"], input_size=DS_SIZES[config.benchmark.factory_args.benchmark_name],
    )

    optimizer, scheduler_plugin = model_factory.get_optimizer(
        model,
        optimizer_type=config.optimizer.type,
        scheduler_type=config.scheduler.type,
        kwargs_optimizer=config["optimizer"],
        kwargs_scheduler=config["scheduler"],
    )
    print(optimizer)

    if scheduler_plugin is not None:
        plugins.append(scheduler_plugin)

    if not config.wandb.run_name:
        config.wandb.run_name = wandb.run.name
    logdir = os.path.join(config.experiment.results_root, config.wandb.run_group, config.wandb.run_name)
    if config.experiment.logdir is None:
        os.makedirs(logdir, exist_ok=True)
        # Add full results dir to config
        config.experiment.logdir = logdir
        omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yaml"))
    else:
        logdir = config.experiment.logdir

    strategy = method_factory.create_strategy(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        logdir=logdir,
        name=config.strategy.name,
        dataset_name=config.benchmark.factory_args.benchmark_name,
        strategy_kwargs=config["strategy"],
        evaluation_kwargs=config["evaluation"],
        wandb_logger=wandb_logger,
        misc=data_dict,
    )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", strategy.plugins)

    has_val_stream = hasattr(scenario, "valid_stream")
    results = dict(task=[], test_accuracy=[])
    batch_streams = scenario.streams.values()
    train_streams, stop_index = scenario.train_stream, config.experiment.stop_at_experience
    start_index, stop_index = 0, len(train_streams) if stop_index is None else stop_index

    for t, experience in enumerate(train_streams[:stop_index]):
        if t < start_index:
            continue  # need this to sync with enumerate

        if config.experiment.train_online:
            ocl_scenario = OnlineCLScenario(
                original_streams=batch_streams,
                experiences=experience,
                experience_size=config.strategy.train_mb_size,
                access_task_boundaries=config.strategy.use_task_boundaries,
            )
            train_stream = ocl_scenario.train_stream
        else:
            train_stream = experience

        strategy.train(
            train_stream,
            eval_streams=[scenario.valid_stream[: t + 1]] if has_val_stream else [],
            num_workers=0,
            drop_last=True,
        )

        if config.experiment.save_models:
            torch.save(strategy.model.state_dict(), os.path.join(logdir, f"model_{t}.ckpt"))

        results["task"].append(t)
        if has_val_stream:
            if "val_accuracy" not in results:
                results["val_accuracy"] = []
            val_result = strategy.eval(scenario.valid_stream[: t + 1])
            results["val_accuracy"].append(val_result['Top1_Acc_Stream/eval_phase/valid_stream/Task000'])

        eval_test_streams = scenario.test_stream[: t + 1]
        test_result = strategy.eval(eval_test_streams)
        results["test_accuracy"].append(test_result['Top1_Acc_Stream/eval_phase/test_stream/Task000'])

    pd.DataFrame(results).to_csv(os.path.join(logdir, f"result.csv"), index=False)
    return results


if __name__ == "__main__":
    utils.preprocess_argv()
    main()
