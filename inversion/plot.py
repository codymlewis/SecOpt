import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sp


def add_image(fn, axis):
    img = mpimg.imread(fn)
    axis.imshow(img)
    axis.axis('off')


def find(files: list, client_id: str, label: str):
    return [f for f in files if f'{client_id}_Z' in f and f[-5] == label][0]


if __name__ == "__main__":
    sgd_files = os.listdir('sgd')
    our_files = os.listdir('ours')
    ground_truths = list(f for f in set(sgd_files) & set(our_files) if 'ground_truth' in f)

    rows, cols = sp.optimize.brute(
        lambda x: np.linalg.norm(np.prod(x.round()) - len(ground_truths)) + np.linalg.norm(x.round()),
        (slice(1, len(ground_truths) / 2), slice(1, len(ground_truths)))
    ).astype(int).tolist()

    for imagename in ['ground_truth', 'sgd', 'ours']:
        fig, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for i, gt in enumerate(ground_truths):
            if imagename == "ground_truth":
                add_image(f"sgd/{gt}", axes[i])
            else:
                client_id = gt[:8]
                label = gt[-5]
                add_image(
                    f"{imagename}/{find(sgd_files if imagename == 'sgd' else our_files, client_id, label)}",
                    axes[i]
                )
        plt.tight_layout(pad=0.1)
        plt.savefig(f'{imagename}_inversions.pdf', dpi=320)
