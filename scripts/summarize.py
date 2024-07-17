import argparse
from functools import reduce
import numpy as np
import os
import pandas as pd
import sys
import yaml

sys.path.insert(0, '..')  # noqa
from src.toolkit.process_results import gather_json_lines


# Example
# python scripts/summarize.py --run_group_dir=results/summarize --group_by strategy.name strategy.train_epochs
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_group_dir', type=str, required=True)
    parser.add_argument('--group_by', type=str, nargs='+', required=True)
    parser.add_argument('--tune_task_index', type=int, default=19)
    return parser.parse_args()


def get_config_val(cfg, key):
    return reduce(lambda c, k: c[k], key.split('.'), cfg)


def compute_AAA(
    dataframe,
    base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000",
):
    """
    Computes average anytime accuracy
    """
    df = dataframe.sort_values(["mb_index"])
    return df[base_name].cumsum()/(df.index+1)


def decorate_with_training_task(
    dataframe,
    num_tasks=20,
    base_name="Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp",
):
    metric_list = [f"{base_name}{i:03d}" for i in range(num_tasks)]
    dataframe["training_exp"] = dataframe[metric_list].count(axis=1)
    return dataframe


def compute_min_acc(dataframe, num_tasks=20):
    df = decorate_with_training_task(dataframe, num_tasks=num_tasks)
    for task in range(num_tasks):
        metric_name = f"Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task:03d}"
        mask = df["training_exp"] - 1 == task
        # df["Min_" + metric_name] = df.groupby("seed", group_keys=False)[
        #     metric_name
        # ].apply(lambda x: x.mask(mask).cummin())
        df["Min_" + metric_name] = df[[metric_name]].apply(lambda x: x.mask(mask).cummin())
    return df


def compute_wcacc(dataframe, num_tasks=20):
    """
    Computes WC-Acc
    """
    df = compute_min_acc(dataframe, num_tasks)
    df = df[~(df.training_exp == 0)]
    df = df.reset_index()

    average_cols = [
        f"Min_Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task:03d}"
        for task in range(num_tasks)
    ]

    exclude_cols = [
        f"Min_Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task-1:03d}"
        for task in df["training_exp"]
    ]

    for i, row in df.iterrows():
        df.loc[i, exclude_cols[i]] = np.nan

    filter_cols = [
        f"Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task-1:03d}"
        for task in df["training_exp"]
    ]

    df["average_min_acc"] = df[average_cols].mean(axis=1)

    for i, row in df.iterrows():
        df.loc[i, "current_task_acc"] = df.loc[i, filter_cols[i]]

    df["WCAcc"] = (
        df["current_task_acc"] * (1 / df["training_exp"])
        + (1 - 1 / df["training_exp"]) * df["average_min_acc"]
    )
    return df


def get_AAA(run_dir) -> pd.DataFrame:
    # HACK: While we have access to AAA from the entire training history, set df's AAA for all tasks to only final AAA.
    path = os.path.join(run_dir, 'logs_continual.json')
    if not os.path.exists(path):
        return None

    new_lines = gather_json_lines(path)
    df_aaa = pd.DataFrame(new_lines)
    # series_aaa = compute_AAA(df_aaa, base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000")
    # df['aa_accuracy'] = series_aaa.iloc[-1]
    return compute_AAA(df_aaa, base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000")


def get_WCACC(run_dir, num_tasks) -> pd.DataFrame:
    # HACK: While we have access to WCACC from the entire training history,
    # set df's WCACC for all tasks to only final WCACC.
    path = os.path.join(run_dir, 'logs_continual.json')
    if not os.path.exists(path):
        return None

    new_lines = gather_json_lines(path)
    df_wcacc = pd.DataFrame(new_lines)
    return compute_wcacc(df_wcacc, num_tasks=num_tasks)['WCAcc']


def main(args):
    run_group_dir = args.run_group_dir
    group_by = args.group_by
    assert os.path.exists(run_group_dir)

    run_dirs, run_configs = [], []
    for run_name in os.listdir(run_group_dir):
        run_dir = os.path.join(run_group_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        with open(os.path.join(run_dir, "config.yaml"), "r") as stream:
            run_config = yaml.safe_load(stream)
        run_dirs.append(run_dir)
        run_configs.append(run_config)

    grouped_run_dirs = dict()
    for run_dir, run_config in zip(run_dirs, run_configs):
        group_key = '|'.join([f"{g_var}:{get_config_val(run_config, g_var)}" for g_var in group_by])
        if group_key in grouped_run_dirs:
            grouped_run_dirs[group_key].append(run_dir)
        else:
            grouped_run_dirs[group_key] = [run_dir]

    all_results = []
    for group_name, run_dirs in grouped_run_dirs.items():
        run_dfs, aaa_length, wcacc_length = [], None, None
        for run_dir in run_dirs:
            run_df = pd.read_csv(os.path.join(run_dir, 'result.csv'))
            series_aaa = get_AAA(run_dir)
            series_wcacc = get_WCACC(run_dir, args.tune_task_index+1)
            if series_aaa is None:
                print(f"WARNING: AAA/WCACC missing for {run_dir}.")
            else:
                if aaa_length is None:
                    aaa_length = len(series_aaa)
                else:
                    curr_length = len(series_aaa)
                    assert aaa_length == curr_length, f"AAA: Correct vs Curr Length: {aaa_length} / {curr_length}"
                if wcacc_length is None:
                    wcacc_length = len(series_wcacc)
                else:
                    curr_length = len(series_wcacc)
                    assert wcacc_length == curr_length, f"WCACC: Correct vs Curr Length: {wcacc_length} / {curr_length}"
                run_df['aa_accuracy'] = series_aaa.iloc[-1]
                run_df['wc_accuracy'] = series_wcacc.iloc[-1]
            run_dfs.append(run_df)

        group_results = pd.concat(run_dfs)
        stat_columns = [col for col in group_results.columns if 'accuracy' in col]
        group_results = group_results.groupby('task', as_index=False)

        mean_df = group_results.mean().rename({k: f"avg_{k}" for k in stat_columns}, axis='columns')
        ste_df = group_results.std().rename({k: f"ste_{k}" for k in stat_columns}, axis='columns')
        for k in stat_columns:
            ste_df[f"ste_{k}"] /= len(run_dfs)**0.5
        group_results = pd.merge(mean_df, ste_df, on='task').fillna(0).round(4)
        group_results["name"] = group_name
        for group_var_and_val in group_name.split('|'):
            group_var, group_val = group_var_and_val.split(':')
            group_results[group_var] = group_val
        all_results.append(group_results)

    # All results
    all_results = pd.concat(all_results).reset_index(drop=True)
    all_results = all_results.rename(columns=lambda n: n.split('.')[-1] if '.' in n else n)
    all_results.drop(columns=['name']).to_csv(os.path.join(run_group_dir, 'summary_all.csv'), index=False)

    # Final task results ordered by test scores
    results = all_results[all_results.task == all_results.task.max()].sort_values('avg_test_accuracy', ascending=False)
    results.drop(columns=['name']).to_csv(os.path.join(run_group_dir, 'summary.csv'), index=False)

    # Final task results ordered by validation scores of args.tune_task_index-th task
    val_ordered = all_results[all_results.task == args.tune_task_index].sort_values('avg_val_accuracy', ascending=False)
    val_ordered_results = pd.concat([results[results.name == name] for name in val_ordered['name'].tolist()])
    val_ordered_results.drop(columns=['name']).to_csv(os.path.join(run_group_dir, 'summary_tune.csv'), index=False)

    print(val_ordered_results.drop(columns=['name']))


if __name__ == "__main__":
    args = parse_args()
    main(args)
