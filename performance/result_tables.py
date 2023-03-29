"""
Some code that outputs a latex table from the csv result data in this repository.
Works for the inversion and perfromance experiment results, mitigation has its
own specific version
"""

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt


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


def format_aggregation_names(agg_names):
    agg_names = agg_names.str.replace("fedavg", "FedAVG")
    agg_names = agg_names.str.replace("^adam", "FedAVG w/ Adam", regex=True)
    agg_names = agg_names.str.replace("oursfedadam", "Ours w/ FedAdam")
    agg_names = agg_names.str.replace("fedadam", "FedAdam")
    agg_names = agg_names.str.replace("secagg", "SecAgg")
    agg_names = agg_names.str.replace("ours", "Ours")
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
    parser = ArgumentParser(description="Process the results file into a respesentative LaTeX table.")
    parser.add_argument('-e', '--efficient', action='store_true', help='Show only the efficient results')
    parser.add_argument('-p', '--plot', action='store_true', help="Plot the results")
    args = parser.parse_args()

    df = pd.read_csv('results.csv').drop(columns=['rounds', 'seed'])
    df = df[df.efficient == args.efficient].drop(columns='efficient')
    grouping_col_set = {'batch_size', 'dataset', 'model', 'num_clients', 'aggregation', 'iid', 'epochs'}

    grouping_col_names = list(set(df.columns) & grouping_col_set)
    groups = df.groupby(grouping_col_names)
    g_mean = groups.mean(numeric_only=True).reset_index()
    g_std = groups.std(numeric_only=True).reset_index()

    if not args.plot:
        for col in g_mean.columns:
            if col not in grouping_col_names:
                if "accuracy" in col.lower() or "asr" in col.lower() or "attack success" in col.lower():
                    g_mean[col] = g_mean[col].map("{:.3%}".format) + g_std[col].map(" ({:.3%})".format)
                elif "rounds for convergence" in col.lower():
                    g_mean[col] = g_mean[col].map("{:.3f}".format) + g_std[col].map(" ({:.3f})".format)
                else:
                    g_mean[col] = g_mean[col].astype(str) + " (" + g_std[col].astype(str) + ")"
    else:
        for col in g_mean.columns:
            if col not in grouping_col_names:
                g_mean[f"{col} Mean"] = g_mean[col]
                g_mean[f"{col} STD"] = g_std[col]
                g_mean = g_mean.drop(columns=col)

    agg_results = g_mean
    agg_results = agg_results.drop(columns=['batch_size', 'num_clients', "convergence"])
    agg_results.model = agg_results.model.pipe(format_model_names)
    agg_results.dataset = agg_results.dataset.pipe(format_dataset_names)
    agg_results.aggregation = agg_results.aggregation.pipe(format_aggregation_names)
    cols = agg_results.columns.tolist()
    ordered_cols = ['aggregation', 'dataset', 'model', 'iid', 'epochs']
    agg_results = agg_results[
        ordered_cols + list(set(cols) - set(ordered_cols))
    ]
    agg_results = agg_results.sort_values(["aggregation", "dataset", "model", 'iid', 'epochs'])
    # agg_results = agg_results.drop(columns=["iid", "epochs"])

    if not args.plot:
        print(agg_results.style.pipe(format_final_table).to_latex(position_float='centering'))
    else:
        data = agg_results[
            (agg_results.dataset == "CIFAR-10") & (agg_results.iid == 0.5) & (agg_results.model == "CNN1")
        ]
        fig, ax = plt.subplots()
        for aggregation in data.aggregation.unique():
            agg_data = data[data.aggregation == aggregation]
            ax.plot(agg_data.epochs, agg_data['Final accuracy Mean'], '-', label=aggregation)
            ax.fill_between(
                agg_data.epochs,
                agg_data['Final accuracy Mean'] - agg_data['Final accuracy STD'],
                agg_data['Final accuracy Mean'] + agg_data['Final accuracy STD'],
                alpha=0.2
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
