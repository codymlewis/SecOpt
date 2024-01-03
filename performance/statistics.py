import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("results.csv")
    df = df.drop(columns=["seed", "clients", "rounds", "epochs", "steps", "batch_size"])
    df = df.groupby(['dataset', 'model', 'pgd', 'server_optimiser', 'client_optimiser']).mean()
    df = df.reset_index()
    print(df.style.to_latex(position_float='centering'))
