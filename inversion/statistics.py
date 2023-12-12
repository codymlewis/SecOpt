import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("results.csv")
    df = df.groupby(['dataset', 'model', 'pgd', 'attack', 'optimiser', 'batch_size']).mean()
    df = df.reset_index()
    print(df.style.to_latex(position_float='centering'))
