import cv2
from pathlib import Path

def process_images_from_directory(directory, label, size):
    """
    Loads and preprocesses images from a specified directory.

    Parameters:
    - directory (Pathlib.Path): Path to the image directory.
    - label (list): List for labels.
    - size (int): Size to which each image is resized.

    Returns:
    - list: A list of processed images.
    """
    images_list = []
    for image_path in directory.iterdir():
        try:
            if image_path.suffix == '.png':
                # Read the image using OpenCV
                image = cv2.imread(str(image_path))

                # Convert the image to RGB (OpenCV uses BGR by default)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize the image
                image_resized = cv2.resize(image_rgb, (size, size))

                images_list.append(image_resized)
                label.append(0 if directory.name == 'Parasitized' else 1)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    return images_list
