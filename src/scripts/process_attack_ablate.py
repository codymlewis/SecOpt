import itertools
import pandas as pd

data = ""
l1_regs = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
l2_regs = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
regs = itertools.cycle(itertools.product(l1_regs, l2_regs))
seed_cycle = itertools.cycle([0, 1, 2, 3, 4])

with open("inversion_results.csv", 'r') as f:
    first = True
    for line in f:
        if first:
            data += f"{line[:-1]},l1_reg,l2_reg\n"
            first = False
        else:
            if next(seed_cycle) == 0:
                l1_reg, l2_reg = next(regs)
            data += f"{line[:-1]},{l1_reg},{l2_reg}\n"

with open("new_inversion_results.csv", 'w') as f:
    f.write(data)

df = pd.read_csv("new_inversion_results.csv")
df = df.drop(columns=["model", "pgd", "attack", "optimiser", "seed", "batch_size", "steps", "regularise"])
for dataset, ddf in iter(df.groupby(["dataset", "l1_reg", "l2_reg"]).mean().reset_index().groupby(["dataset"])):
    max_ssim = ddf["ssim"].max()
    print(ddf.query("`ssim` == @max_ssim"))
