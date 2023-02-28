"""
Some code that outputs a latex table from the csv result data in this repository.
Works for the inversion and perfromance experiment results, mitigation has its
own specific version
"""

import pandas as pd
from argparse import ArgumentParser


def format_model_names(model_names):
    model_names = model_names.str.replace('lenet', 'LeNet')
    model_names = model_names.str.replace('cnn1', 'CNN1')
    model_names = model_names.str.replace('cnn2', 'CNN2')
    model_names = model_names.str.replace('resnet', 'ResNet18 V2')
    return model_names


def format_dataset_names(dataset_names):
    dataset_names = dataset_names.str.replace('mnist', 'FMNIST')
    dataset_names = dataset_names.str.replace('cifar10', 'CIFAR-10')
    dataset_names = dataset_names.str.replace('svhn', 'SVHN')
    return dataset_names


def format_opt_names(opt_names):
    opt_names = opt_names.str.replace('sgd', 'SGD')
    opt_names = opt_names.str.replace('adam', 'Adam')
    opt_names = opt_names.str.replace('ours', 'Ours')
    return opt_names


def format_aggregation_names(agg_names):
    agg_names = agg_names.str.replace("fedavg", "FedAVG")
    agg_names = agg_names.str.replace("secagg", "SecAgg")
    agg_names = agg_names.str.replace("nerv", "Ours")
    return agg_names


def format_final_table(styler):
    columns = styler.columns
    acc_asr_cols = set(
        str(col) for col in columns
        if 'accuracy' in col.lower() or 'asr' in col.lower() or 'attack success' in col.lower()
    )
    other_columns = set(str(col) for col in columns) - acc_asr_cols
    styler = styler.format(formatter=lambda s: s.replace('%', '\\%'), subset=list(acc_asr_cols))
    styler = styler.format(precision=3, subset=list(other_columns))
    styler = styler.hide()
    return styler


if __name__ == "__main__":
    parser = ArgumentParser(description="Transform an experiment results csv into a LaTeX table")
    parser.add_argument('--dp', action="store_true", help="Parse differential privacy experiment results.")
    parser.add_argument('--idlg', action="store_true", help="Parse iDLG experiment results.")
    args = parser.parse_args()

    df = pd.read_csv('results.csv').drop(columns=['rounds', 'seed'])
    if args.dp:
        df = df.where(df.clipping_rate > 0).dropna()
        df = df.drop(columns="opt")
        grouping_col_set = {
            'batch_size', 'dataset', 'model', 'num_clients', 'aggregation', 'clipping_rate', 'noise_scale'
        }
    else:
        if "clipping_rate" in df.columns:
            df = df.where(df.clipping_rate == 0).dropna()
            df = df.drop(columns=["clipping_rate", "noise_scale"])
        grouping_col_set = {'batch_size', 'dataset', 'model', 'num_clients', 'opt', 'aggregation'}

    if args.idlg:
        df = df.where(df.batch_size == 1).dropna()
    else:
        df = df.where(df.batch_size > 1).dropna()

    grouping_col_names = list(set(df.columns) & grouping_col_set)
    groups = df.groupby(grouping_col_names)
    g_mean = groups.mean().reset_index()
    g_std = groups.std().reset_index()
    for col in g_mean.columns:
        if col not in grouping_col_names:
            if "accuracy" in col.lower() or 'asr' in col.lower() or 'attack success' in col.lower():
                g_mean[col] = g_mean[col].map("{:.3%}".format) + g_std[col].map(" ({:.3%})".format)
            elif "psnr" in col.lower() or "ssim" in col.lower() or "cm" in col.lower():
                g_mean[col] = g_mean[col].map("{:.3f}".format) + g_std[col].map(" ({:.3f})".format)
            else:
                g_mean[col] = g_mean[col].astype(str) + " (" + g_std[col].astype(str) + ")"
    agg_results = g_mean
    agg_results = agg_results.drop(columns=['batch_size', 'num_clients'])
    agg_results.model = agg_results.model.pipe(format_model_names)
    agg_results.dataset = agg_results.dataset.pipe(format_dataset_names)
    if 'opt' in agg_results.columns:
        agg_results.opt = agg_results.opt.pipe(format_opt_names)
        cols = agg_results.columns.tolist()
        agg_results = agg_results[
            ["opt", "dataset", "model", "Final accuracy"] + list(set(cols) - {"opt", "dataset", "model", "Final accuracy"})
        ]
        agg_results = agg_results.sort_values(["opt", "dataset", "model"])
    if 'aggregation' in agg_results.columns:
        agg_results.aggregation = agg_results.aggregation.pipe(format_aggregation_names)
        cols = agg_results.columns.tolist()
        agg_results = agg_results[
            ["aggregation", "dataset", "model"] + list(set(cols) - {"aggregation", "dataset", "model"})
        ]
        agg_results = agg_results.sort_values(["aggregation", "dataset", "model"])
    if args.dp:
        cols = agg_results.columns.tolist()
        first_cols = ["dataset", "model", "clipping_rate", "noise_scale"]
        other_cols = list(set(cols) - set(first_cols))
        other_cols.sort()
        agg_results = agg_results[first_cols + other_cols]
        agg_results = agg_results.sort_values(first_cols)
    print(agg_results.style.pipe(format_final_table).to_latex(position_float='centering'))
