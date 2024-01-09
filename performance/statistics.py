import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("results.csv")
    df = df.drop(columns=[
        "seed",
        "clients",
        "rounds",
        "epochs",
        "steps",
        "batch_size",
        "participation_rate",
        "client_learning_rate",
        "server_learning_rate",
    ])
    df = df.groupby([
        'dataset', 'model', 'pgd', 'server_optimiser', 'client_optimiser', 'clip_threshold', 'noise_scale'
    ]).mean()
    df = df.reset_index()
    print(df.style.hide().to_latex(position_float='centering'))
