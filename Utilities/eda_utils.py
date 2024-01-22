"""Module for performing exploratory data analysis (EDA) on image datasets."""

from import_libraries import get_data_handling_and_viz_libs
np, plt, sns, cv2 = get_data_handling_and_viz_libs()

INFECTED = 0
UNINFECTED = 1

def display_image_samples(images, labels, sample_size=10):
    """
    Display a sample of images with labels.

    Args:
        images (list): List of images.
        labels (list): Corresponding labels for the images.
        sample_size (int): Number of samples to display. Defaults to 10.

    Raises:
        ValueError: If sample_size is larger than the number of available images.
    """
    if sample_size > len(images):
        raise ValueError("sample_size is larger than the available number of images.")

    sample_indices = np.random.choice(len(images), sample_size, replace=False)
    sample_images = [images[i] for i in sample_indices]
    sample_labels = [labels[i] for i in sample_indices]

    plt.figure(figsize=(12, 6))
    for index, (image, label) in enumerate(zip(sample_images, sample_labels)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image)
        plt.title('Infected' if label == INFECTED else 'Uninfected')
        plt.axis('off')
    plt.suptitle("Sample Images")
    plt.show()

def plot_class_distribution(labels):
    """
    Plot the class distribution in the dataset.

    Args:
        labels (list): List of labels in the dataset.
        title (str): Title for the plot.
    """
    classes, counts = np.unique(labels, return_counts=True)
    plt.bar(classes, counts, color=['#800080', '#708090'], width=0.6)
    plt.title("Class Distribution")
    plt.ylabel('Number of samples')
    plt.xticks(classes, ['Infected', 'Uninfected'])

    # Adding count annotations on each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 3, str(count), ha = 'center', va = 'bottom')

    plt.show()

