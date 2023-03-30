import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def add_image(fn, axis):
    img = mpimg.imread(fn)
    axis.imshow(img)
    axis.axis('off')


if __name__ == "__main__":
    adam_files = os.listdir('adam')
    our_files = os.listdir('ours')
    image_fns = list(f for f in set(adam_files) & set(our_files))

    fig, axes = plt.subplots(5, 2)

    for i, fn in enumerate(image_fns[:5]):
        add_image(f"adam/{fn}", axes[i, 0])
        add_image(f"ours/{fn}", axes[i, 1])

    plt.tight_layout(pad=0.1)
    plt.savefig('gradcams.pdf', dpi=320)
