"""
Some code that outputs a latex table from the csv result data in this repository.
"""

import pandas as pd


def format_model_names(model_names):
    model_names = model_names.str.replace('lenet', 'LeNet')
    model_names = model_names.str.replace('cnn1', 'CNN1')
    model_names = model_names.str.replace('cnn2', 'CNN2')
    return model_names

def format_dataset_names(dataset_names):
    dataset_names = dataset_names.str.replace('mnist', 'MNIST')
    dataset_names = dataset_names.str.replace('cifar10', 'CIFAR-10')
    dataset_names = dataset_names.str.replace('svhn', 'SVHN')
    return dataset_names


def format_opt_names(opt_names):
    opt_names = opt_names.str.replace('sgd', 'SGD')
    opt_names = opt_names.str.replace('adam', 'Adam')
    return opt_names


def format_final_table(styler):
    columns = styler.columns
    acc_asr_cols = set(
        str(col) for col in columns
        if 'accuracy' in col.lower() or 'asr' in col.lower() or 'attack success' in col.lower()
    )
    other_columns = set(str(col) for col in columns) - acc_asr_cols
    styler = styler.format(formatter=lambda s: f"{s:.3%}".replace('%', '\\%'), subset=list(acc_asr_cols))
    styler = styler.format(precision=3, subset=list(other_columns))
    styler = styler.hide()
    return styler


if __name__ == "__main__":
    df = pd.read_csv('results.csv').drop(columns=['rounds', 'seed'])
    grouping_col_names = list(set(df.columns) & {'batch_size', 'dataset', 'model', 'num_clients', 'opt'}) 
    groups = df.groupby(grouping_col_names)
    g_mean = groups.mean().reset_index()
    g_std = groups.std().reset_index()
    g_mean.columns = [col + ' (mean)' if col not in grouping_col_names else str(col) for col in g_mean.columns]
    g_std.columns = [col + ' (std)' if col not in grouping_col_names else col for col in g_std.columns]
    agg_results = g_mean.merge(g_std)
    agg_results = agg_results.drop(columns=['batch_size', 'num_clients'])
    agg_results.model = agg_results.model.pipe(format_model_names)
    agg_results.dataset = agg_results.dataset.pipe(format_dataset_names)
    if 'opt' in agg_results.columns:
        agg_results.opt = agg_results.opt.pipe(format_opt_names)
    print(agg_results.style.pipe(format_final_table).to_latex(position_float='centering'))
