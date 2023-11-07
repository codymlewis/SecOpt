from typing import Iterable
import itertools
import pandas as pd
import scipy.stats as sps


def limited_query(df, **kwargs):
    return df.query(' and '.join([f"`{k}` == '{v}'" for k, v in kwargs.items()]))


def analysis_results(df, **kwargs):
    exp_variables = []
    k_to_remove = []
    for k, v in kwargs.items():
        if isinstance(v, Iterable) and not isinstance(v, str):
            exp_variables.append([{k: sub_v} for sub_v in v])
            k_to_remove.append(k)
    for k in k_to_remove:
        del kwargs[k]
    results = []
    for ev in itertools.product(*exp_variables):
        exp_consts = kwargs.copy()
        for v in ev:
            exp_consts.update(v)
        filtered_results = limited_query(df, **exp_consts)
        if len(filtered_results) > 0:
            results.append(filtered_results)
            print(f"Summary statistics for: {ev}")
            print(results[-1].describe())
            print()
    print(f"ANOVA results: {sps.f_oneway(*[r.ssim for r in results])}")
    concat_results = pd.concat(results)
    print("Pearson correlation between the accuracy and attack SSIM: {}".format(
        sps.pearsonr(concat_results.accuracy, concat_results.ssim)
    ))


if __name__ == "__main__":
    df = pd.read_csv("ablation_results.csv")
    print(f"Pearson correlation between the accuracy and attack SSIM: {sps.pearsonr(df.accuracy, df.ssim)}")
    print(f"Pearson correlation between the accuracy and attack PSNR: {sps.pearsonr(df.accuracy, df.psnr)}")
    print(f"Pearson correlation between the attack SSIM and attack PSNR: {sps.pearsonr(df.ssim, df.psnr)}")

    print("Activation analysis")
    analysis_results(df, activation=pd.unique(df.activation).tolist(), pooling="none", normalisation="none")
    print()
    print("Pooling analysis")
    analysis_results(
        df, activation="relu", pooling=pd.unique(df.pooling), pool_size="small", normalisation="none"
    )
    print()
    print("Pooling with window size analysis")
    analysis_results(
        df, activation="relu", pooling=pd.unique(df.pooling), pool_size=pd.unique(df.pool_size), normalisation="none"
    )
    print()
    print("Normalisation analysis")
    analysis_results(df, activation="relu", pooling="none", normalisation=pd.unique(df.normalisation))
    print()
    print("Activation + pooling analysis")
    analysis_results(
        df,
        activation=pd.unique(df.activation),
        pooling=pd.unique(df.pooling),
        pool_size=pd.unique(df.pool_size),
        normalisation="none"
    )
    print()
    print("Activation + normalisation analysis")
    analysis_results(
        df, activation=pd.unique(df.activation), pooling="none", pool_size="small", normalisation=pd.unique(df.normalisation)
    )
    print()
    print("Pooling + normalisation analysis")
    analysis_results(
        df,
        activation="relu",
        pooling=pd.unique(df.pooling),
        pool_size=pd.unique(df.pool_size),
        normalisation=pd.unique(df.normalisation)
    )
    print()
    print("All analysis")
    analysis_results(
        df,
        activation=pd.unique(df.activation),
        pooling=pd.unique(df.pooling),
        pool_size=pd.unique(df.pool_size),
        normalisation=pd.unique(df.normalisation)
    )
    print()
