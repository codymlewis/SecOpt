from scipy import stats
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("ablation_results.csv")
    act_vals = []
    for act in df['activation'].unique():
        act_vals.append(df[df['activation'] == act]['ssim'].to_numpy())
        print(f"Activation {act}, mean SSIM {act_vals[-1].mean()}")
    print(stats.f_oneway(*act_vals))
