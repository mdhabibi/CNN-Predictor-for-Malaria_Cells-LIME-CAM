# Standard library imports
from pathlib import Path  # For handling filesystem paths

import os  # For operating system dependent functionality

# Third-party imports for data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
import cv2  # For image processing

# TensorFlow and Keras for deep learning
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# Scikit-learn for data preprocessing and model evaluation
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # To convert a class vector (integers) to a binary class matrix
from sklearn.metrics import classification_report, confusion_matrix

# Keras Tuner Notebook
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
