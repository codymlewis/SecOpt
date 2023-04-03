import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np


def add_image(fn, axis):
    img = mpimg.imread(fn)
    axis.imshow(img)
    axis.axis('off')


if __name__ == "__main__":
    adam_files = os.listdir('adam')
    our_files = os.listdir('ours')
    image_fns = list(f for f in set(adam_files) & set(our_files))

    rows, cols = sp.optimize.brute(
        lambda x: np.linalg.norm(np.prod(x.round()) - len(image_fns)) + np.linalg.norm(x.round()),
        (slice(1, len(image_fns) / 2), slice(1, len(image_fns)))
    ).astype(int).tolist()

    for opt in ["ground_truth", "adam", "ours"]:
        fig, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for i, fn in enumerate(image_fns):
            add_image(f"{opt}/{fn}", axes[i])
        plt.tight_layout(pad=0.1)
        plt.savefig(f'{opt}_gradcams.pdf', dpi=320)
