"""
Configuration file for Alzheimer's Disease Stage Classification
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_images')

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Classes
CLASSES = ['Non Demented', 'Moderate Dementia']
NUM_CLASSES = len(CLASSES)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATES = [0.001, 0.01]
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Feature extraction parameters
DWT_WAVELET = 'haar'
DWT_LEVEL = 3
LBP_RADIUS = 3
LBP_POINTS = 24
PCA_COMPONENTS = 100

# Model parameters
DROPOUT_RATE = 0.5
DENSE_UNITS = [512, 256, 128]

# Random seed for reproducibility
RANDOM_SEED = 42
