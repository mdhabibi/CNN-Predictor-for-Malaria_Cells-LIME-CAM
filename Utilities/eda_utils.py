import matplotlib.pyplot as plt
import numpy as np

def display_samples(images, labels, title, sample_size=10):
    """Display a sample of images with labels.

    Args:
        images (list): List of images.
        labels (list): Corresponding labels for the images.
        title (str): Title for the plot.
        sample_size (int): Number of samples to display.
    """
    sample_indices = np.random.choice(len(images), sample_size, replace=False)
    sample_images = [images[i] for i in sample_indices]
    sample_labels = [labels[i] for i in sample_indices]

    plt.figure(figsize=(12, 6))
    for index, (image, label) in enumerate(zip(sample_images, sample_labels)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image)
        plt.title('Infected' if label == 0 else 'Uninfected')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_class_distribution(labels, title):
    """Plot the class distribution in the dataset.

    Args:
        labels (list): List of labels in the dataset.
        title (str): Title for the plot.
    """
    classes, counts = np.unique(labels, return_counts=True)
    plt.bar(classes, counts)
    plt.title(title)
    plt.ylabel('Number of samples')
    plt.xticks(classes, ['Parasitized', 'Uninfected'])
    plt.show()

