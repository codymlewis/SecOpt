import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

    # fig, axes = plt.subplots(len(ground_truths), 3)
    fig, axes = plt.subplots(5, 3)
    
    for i, gt in enumerate(ground_truths[:5]):
        add_image(f"sgd/{gt}", axes[i, 0])
        client_id = gt[:8]
        label = gt[-5]
        add_image(f"sgd/{find(sgd_files, client_id, label)}", axes[i, 1])
        add_image(f"ours/{find(our_files, client_id, label)}", axes[i, 2])

    plt.tight_layout(pad=0.1)
    plt.savefig('inversions.pdf', dpi=320)
