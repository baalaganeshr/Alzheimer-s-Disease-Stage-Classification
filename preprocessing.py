"""
Preprocessing pipeline for Alzheimer's Disease Classification
Includes: Median filtering, skull stripping, min-max normalization, resizing
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
import config


class ImagePreprocessor:
    def __init__(self):
        self.img_height = config.IMG_HEIGHT
        self.img_width = config.IMG_WIDTH
    
    def median_filter(self, image, kernel_size=5):
        """
        Apply median filtering for noise removal
        """
        if len(image.shape) == 3:
            return cv2.medianBlur(image, kernel_size)
        else:
            return ndimage.median_filter(image, size=kernel_size)
    
    def skull_stripping(self, image):
        """
        Skull stripping to isolate brain regions
        Uses thresholding and morphological operations
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Otsu's thresholding
        thresh_val = filters.threshold_otsu(gray)
        binary = gray > thresh_val
        
        # Morphological operations to remove noise
        binary = morphology.remove_small_objects(binary, min_size=500)
        binary = morphology.remove_small_holes(binary, area_threshold=500)
        
        # Apply erosion and dilation
        kernel = morphology.disk(3)
        binary = morphology.erosion(binary, kernel)
        binary = morphology.dilation(binary, kernel)
        
        # Create mask for original image
        if len(image.shape) == 3:
            mask = np.stack([binary] * 3, axis=-1)
        else:
            mask = binary
        
        # Apply mask
        stripped = image * mask.astype(image.dtype)
        
        return stripped.astype(np.uint8)
    
    def min_max_normalization(self, image):
        """
        Min-max intensity normalization to [0, 1] range
        """
        img_float = image.astype(np.float32)
        img_min = img_float.min()
        img_max = img_float.max()
        
        if img_max - img_min > 0:
            normalized = (img_float - img_min) / (img_max - img_min)
        else:
            normalized = img_float
        
        return normalized
    
    def resize_image(self, image):
        """
        Resize image to uniform dimensions
        """
        resized = cv2.resize(image, (self.img_width, self.img_height), 
                           interpolation=cv2.INTER_AREA)
        return resized
    
    def preprocess(self, image, save_steps=False):
        """
        Complete preprocessing pipeline
        Returns: preprocessed image and intermediate steps (if save_steps=True)
        """
        steps = {'original': image.copy()} if save_steps else {}
        
        # Step 1: Median filtering
        filtered = self.median_filter(image)
        if save_steps:
            steps['median_filtered'] = filtered.copy()
        
        # Step 2: Skull stripping
        stripped = self.skull_stripping(filtered)
        if save_steps:
            steps['skull_stripped'] = stripped.copy()
        
        # Step 3: Min-max normalization
        normalized = self.min_max_normalization(stripped)
        if save_steps:
            steps['normalized'] = (normalized * 255).astype(np.uint8)
        
        # Step 4: Resizing
        resized = self.resize_image(normalized)
        if save_steps:
            steps['resized'] = (resized * 255).astype(np.uint8)
        
        return resized, steps if save_steps else resized


def preprocess_dataset(image_paths, labels, preprocessor=None, save_steps=False):
    """
    Preprocess entire dataset
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()
    
    preprocessed_images = []
    all_steps = [] if save_steps else None
    
    for img_path in image_paths:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Preprocess
        if save_steps:
            preprocessed, steps = preprocessor.preprocess(image, save_steps=True)
            all_steps.append(steps)
        else:
            preprocessed = preprocessor.preprocess(image, save_steps=False)
        
        preprocessed_images.append(preprocessed)
    
    preprocessed_images = np.array(preprocessed_images)
    labels = np.array(labels)
    
    if save_steps:
        return preprocessed_images, labels, all_steps
    return preprocessed_images, labels


if __name__ == "__main__":
    # Test preprocessing
    import os
    import glob
    
    # Get sample images
    sample_path = glob.glob(os.path.join(config.DATASET_PATH, '*', '*.jpg'))[:1]
    
    if sample_path:
        preprocessor = ImagePreprocessor()
        img = cv2.imread(sample_path[0])
        preprocessed, steps = preprocessor.preprocess(img, save_steps=True)
        
        print("Preprocessing completed successfully!")
        print(f"Final shape: {preprocessed.shape}")
    else:
        print("No images found in dataset path")
