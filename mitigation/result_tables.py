"""
Some code that outputs a latex table from the csv result data.
"""

import pandas as pd
from argparse import ArgumentParser


def format_model_names(model_names):
    model_names = model_names.str.replace('lenet', 'LeNet')
    model_names = model_names.str.replace('cnn1', 'CNN1')
    model_names = model_names.str.replace('cnn2', 'CNN2')
    return model_names

def format_dataset_names(dataset_names):
    dataset_names = dataset_names.str.replace('mnist', 'FMNIST')
    dataset_names = dataset_names.str.replace('cifar10', 'CIFAR-10')
    dataset_names = dataset_names.str.replace('svhn', 'SVHN')
    return dataset_names


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
    parser = ArgumentParser(description="Process the experiment results")
    parser.add_argument('--pgd', action='store_true',
                        help="Display the PGD data or other data [Default: other data]")
    parser.add_argument('--one-shot', action='store_true',
                        help="Display the one shot attack data or continuous attack data [Default: continuous attack data]")
    parser.add_argument('--noise-clip', action='store_true',
                        help="Display the noising and clipping defense data or other data [Default: other data]")
    args = parser.parse_args()

    df = pd.read_csv('results.csv').drop(columns=['rounds', 'seed'])
    grouping_col_names = ['batch_size', 'dataset', 'model', 'num_clients', 'start_round', 'epochs']
    if args.pgd:
        df = df.where(df.hardening == "pgd").dropna()
        grouping_col_names += ['eps']
    else:
        df = df.where(df.hardening != "pgd").dropna()
        df = df.where(df.noise_clip == args.noise_clip).dropna()
        df = df.drop(columns='eps')
    df = df.drop(columns=["noise_clip", "hardening"])
    df = df.where(df.one_shot == args.one_shot).dropna()
    df = df.drop(columns="one_shot")
    groups = df.groupby(grouping_col_names)
    g_mean = groups.mean().reset_index()
    g_std = groups.std().reset_index()
    for col in g_mean.columns:
        if col not in grouping_col_names:
            if "accuracy" in col.lower() or 'asr' in col.lower() or 'attack success' in col.lower():
                g_mean[col] = g_mean[col].map("{:.3%}".format) + g_std[col].map(" ({:.3%})".format)
            elif "recovery rounds" in col.lower():
                g_mean[col] = g_mean[col].map("{:.3f}".format) + g_std[col].map(" ({:.3f})".format)
            else:
                g_mean[col] = g_mean[col].astype(str) + " (" + g_std[col].astype(str) + ")"
    agg_results = g_mean
    agg_results = agg_results.drop(columns=['batch_size', 'num_clients', "start_round", "epochs"])
    agg_results.model = agg_results.model.pipe(format_model_names)
    agg_results.dataset = agg_results.dataset.pipe(format_dataset_names)
    cols = agg_results.columns.tolist()
    if args.pgd:
        first_cols = ["dataset", "model", "eps"]
    else:
        first_cols = ["dataset", "model"]
    other_cols = list(set(cols) - set(first_cols))
    other_cols.sort(key=lambda s: s.replace("First at", "Final ar"))  # a bit goofy but it puts the ASRs in the right order
    agg_results = agg_results[first_cols + other_cols]
    agg_results = agg_results.sort_values(first_cols)
    print(agg_results.style.pipe(format_final_table).to_latex(position_float='centering'))
