import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import random


def plot_images_with_labels(filenames, labels, gt, num_images_to_show=16):
    """
    Plot a grid of images with their corresponding labels and ground truth labels.
    :param filenames: list of image filenames
    :param labels: list of predicted image labels
    :param gt: list of ground truth image labels
    :param num_images_to_show: number of images to show
    """
    # Ensure the number of images to show is not greater than the total number of images
    num_images_to_show = min(num_images_to_show, len(filenames))

    # Randomly select indices for the images to display
    selected_indices = random.sample(range(len(filenames)), num_images_to_show)

    # Create a subplot grid
    rows = int(num_images_to_show ** 0.5)
    cols = (num_images_to_show + rows - 1) // rows
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.4)  # Increase the hspace value for more space between rows

    # Plot selected images with labels and ground truth labels
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indices):
        ax = plt.subplot(gs[i])
        img = mpimg.imread(filenames[idx])
        plt.imshow(img)
        plt.axis('off')
        ax.set_title(f"GT: {gt[idx]}\nPredicted: {labels[idx]}", fontsize=8)  # Set the font size to 8

    plt.show()
