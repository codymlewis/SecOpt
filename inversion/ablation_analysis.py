import pandas as pd
import scipy.stats as sps


def limited_query(df, **kwargs):
    return df.query(' and '.join([f"`{k}` == '{v}'" for k, v in kwargs.items()]))


def analysis_results(df, activation="relu", pooling="none", pool_size="small", normalisation="none"):
    pass
    # subresults = [
    #     limited_query(df, )
    # ]


if __name__ == "__main__":
    df = pd.read_csv("ablation_results.csv")
    print(f"Pearson correlation between the accuracy and attack SSIM: {sps.pearsonr(df.accuracy, df.ssim)}")
    print(f"Pearson correlation between the accuracy and attack PSNR: {sps.pearsonr(df.accuracy, df.psnr)}")
    print(f"Pearson correlation between the attack SSIM and attack PSNR: {sps.pearsonr(df.ssim, df.psnr)}")

    activation_results = [
        limited_query(df, activation=a, pooling="none", pool_size="small", normalisation="none")
        for a in pd.unique(df.activation)
    ]
    print(f"Activation ANOVA results: {sps.f_oneway(*[ar.ssim for ar in activation_results])}")
    print("Activation summary statistics:")
    for ar in activation_results:
        print(("=" * 10) + f" {ar.activation.iloc[0]} " + ("=" * 10))
        print(ar.describe())
        print("=" * 30)
    print("Activation ANOVA results: {}".format(
        sps.f_oneway(*[ar.ssim for ar in activation_results if (ar.activation != 'sigmoid').any()])
    ))

    pooling_results = [
        limited_query(df, activation="relu", pooling=p, pool_size="small", normalisation="none")
        for p in pd.unique(df.pooling)
    ]
    print(f"Activation ANOVA results: {sps.f_oneway(*[ar.ssim for ar in pooling_results])}")
