"""Module for importing commonly used libraries and functions."""

def get_general_utils():
    """
    Import and return general utility libraries commonly used for file and system operations.

    Returns:
        Path: A class for representing filesystem paths with semantics appropriate for different operating systems.
        os: A module providing a portable way of using operating system-dependent functionality like reading or writing to the file system, manipulating paths, etc.
    """
    from pathlib import Path
    import os
    return Path, os


def get_data_handling_and_viz_libs():
    """
    Import and return essential libraries used for data handling and visualization in data science and machine learning projects.

    Returns:
        np: The NumPy module, for numerical operations.
        plt: The Matplotlib Pyplot module, for plotting and visualization.
        sns: The Seaborn module, for making attractive and informative statistical graphics.
        cv2: The OpenCV module, for image processing tasks.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import cv2  
    return np, plt, sns, cv2

    
def get_sklearn_components():
    """
    Import and return Scikit-learn components for model selection and evaluation.

    Returns:
        train_test_split: Function for splitting dataset into training and test sets.
        classification_report: Function to build a text report showing the main classification metrics.
        confusion_matrix: Function to compute the confusion matrix to evaluate the accuracy of a classification.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    return train_test_split, classification_report, confusion_matrix, roc_curve, auc

def get_keras_utilities():
    """
    Import and return Keras utility functions.

    Returns:
        to_categorical: Converts a class vector (integers) to binary class matrix, useful for converting labels to a format suitable for training neural networks.
    """
    from keras.utils import to_categorical
    return to_categorical
   
def get_core_keras_layers():
    """
    Import and return core TensorFlow and Keras layers commonly used in building deep learning models.

    Returns:
    	Input: Used to instantiate a Keras tensor.
    	Conv2D: Implements the convolutional layer for spatial convolution over images.
    	Dense: Regular densely-connected neural network layer.
    	Flatten: Flattens the input without affecting the batch size.
    	BatchNormalization: Applies batch normalization, stabilizing and accelerating the training process.
    	Dropout: Applies dropout to the input, helping prevent overfitting in neural networks.
    	Model: Groups layers into an object with training and inference features.
    """
    from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    return Input, Conv2D, Dense, Flatten, BatchNormalization, Dropout, Model

    
def get_training_components():
    """
    Import and return key TensorFlow and Keras components used in the training process of neural networks.

    Returns:
    	l2: L2 regularization function, used to apply a penalty on layer parameters or layer activity during optimization, helping to prevent overfitting.
    	EarlyStopping: Callback to stop training when a monitored metric has stopped improving, aiding in preventing overfitting and saving computational resources.
    	ModelCheckpoint: Callback to save the model or model weights at specified intervals, so the model can be loaded from a checkpoint if the training process is interrupted or we wish to use the model as of a certain epoch.
    	ReduceLROnPlateau: Callback to reduce the learning rate when a metric has stopped improving, which can lead to better training results and model convergence.
		TensorBoard: TensorFlow's visualization toolkit that provides a suite of web applications for inspecting and understanding the TensorFlow runs and graphs.
    """
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    return l2, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def get_data_preprocessing_tools():
    """
    Import and return the ImageDataGenerator class from Keras for image data preprocessing.

    Returns:
        ImageDataGenerator: Class from Keras used for real-time data augmentation and preprocessing of image data.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator
    
def get_lime_related_libs():
    """
    Import and return libraries and functions that we used in LIME_test.ipynb

    Returns:
        skimage.io: Module from scikit-image for reading and writing images.
        skimage.segmentation: Module from scikit-image for image segmentation.
        sklearn: Scikit-learn module for machine learning algorithms.
        LinearRegression: Linear regression model from Scikit-learn.
        warnings: Module to handle warnings.
        load_model: Function from Keras to load pre-trained models.
    """
    #import skimage.io
    #import skimage.segmentation
    import skimage

    import sklearn
    from sklearn.linear_model import LinearRegression
    import warnings
    from keras.models import load_model
    return skimage, sklearn, LinearRegression, warnings, load_model
    
def get_gap_test_and_localization_libs():
    """
    Import and return libraries and functions specifically needed for performing a GAP test
    and localizing anomalies in images with a CNN.

    Returns:
        GlobalAveragePooling2D: The GlobalAveragePooling2D layer for spatial data.
        scipy: The SciPy library, used here for image upsampling.
        Rectangle: The matplotlib class for adding rectangle overlays to images.
        peak_local_max: Function from skimage to detect hotspots in 2D images.
    """
    from tensorflow.keras.layers import GlobalAveragePooling2D
    import scipy
    from matplotlib.patches import Rectangle
    from skimage.feature import peak_local_max

    return GlobalAveragePooling2D, scipy, Rectangle, peak_local_max



if __name__ == "__main__":
    # Test imports
    print("All libraries imported successfully.")
