"""
This module contains functions for processing and loading image data, specifically designed for image classification tasks.
It includes utilities for assigning labels to images based on directory names and for loading and preprocessing images from a given directory.
"""

from import_libraries import get_data_handling_and_viz_libs
np, plt, sns, cv2 = get_data_handling_and_viz_libs()

def assign_label(image_path, positive_class='Parasitized'):
    """
    Assigns a label to an image based on its directory name.

    Parameters:
    - image_path: Path to the image.
    - positive_class (str): Directory name of the positive class.

    Returns:
    - int: Label for the image (0 for infected class, 1 for Uninfected).
    """
    return 0 if image_path.parent.name == positive_class else 1

def load_and_preprocess_images(directory, size, color_mode='rgb', image_formats={'.png', '.jpg', '.jpeg'}):
    """
    Loads and preprocesses images from a specified directory.

    Parameters:
    - directory (Pathlib.Path): Path to the image directory.
    - size (int): Size to which each image is resized.
    - color_mode (str): Color conversion mode ('rgb' or 'grayscale').
    - image_formats (set): Allowed image file formats.

    Returns:
    - list: A list of processed images.
    - list: Corresponding labels of the images.
    """
    images_list = []
    labels_list = []
    for image_path in directory.iterdir():
        try:
            if image_path.suffix.lower() in image_formats:
                # Read the image using OpenCV
                image = cv2.imread(str(image_path))

                # Convert the image based on the specified color mode
                if color_mode == 'rgb':
                    image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif color_mode == 'grayscale':
                    image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Resize the image
                image_resized = cv2.resize(image_processed, (size, size))
                
                # Normalize the image
                image_normalized = image_resized.astype('float32') / 255.

                images_list.append(image_normalized)
                labels_list.append(assign_label(image_path, positive_class='Parasitized'))
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    return images_list, labels_list
