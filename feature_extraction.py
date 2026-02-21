"""
Feature Extraction Module
- DWT (Discrete Wavelet Transform) for MRI
- LBP (Local Binary Pattern) for PET  
- DenseNet for CT
"""
import numpy as np
import pywt
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import config


class DWTFeatureExtractor:
    """
    Discrete Wavelet Transform for multiresolution spatial features (MRI)
    """
    def __init__(self, wavelet=config.DWT_WAVELET, level=config.DWT_LEVEL):
        self.wavelet = wavelet
        self.level = level
    
    def extract(self, image):
        """
        Extract DWT features from image
        Returns flattened feature vector
        """
        features = []
        
        # Handle grayscale or color images
        if len(image.shape) == 3:
            # Process each channel
            for channel in range(image.shape[2]):
                coeffs = pywt.wavedec2(image[:,:,channel], self.wavelet, level=self.level)
                
                # Extract approximation and detail coefficients
                cA = coeffs[0]  # Approximation coefficients
                features.extend(cA.flatten())
                
                # Detail coefficients
                for detail_level in coeffs[1:]:
                    for detail in detail_level:
                        features.extend(detail.flatten())
        else:
            # Grayscale image
            coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
            cA = coeffs[0]
            features.extend(cA.flatten())
            
            for detail_level in coeffs[1:]:
                for detail in detail_level:
                    features.extend(detail.flatten())
        
        return np.array(features)


class LBPFeatureExtractor:
    """
    Local Binary Pattern for texture features (PET)
    """
    def __init__(self, radius=config.LBP_RADIUS, points=config.LBP_POINTS):
        self.radius = radius
        self.points = points
        self.method = 'uniform'
    
    def extract(self, image):
        """
        Extract LBP features from image
        Returns histogram of LBP patterns
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Compute LBP
        lbp = local_binary_pattern(gray, self.points, self.radius, method=self.method)
        
        # Compute histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist


class DenseNetFeatureExtractor:
    """
    DenseNet for high-level anatomical features (CT)
    """
    def __init__(self):
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        self.model = base_model
    
    def extract(self, image):
        """
        Extract DenseNet features from image
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Ensure correct shape (add batch dimension)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Preprocess for DenseNet
        image_preprocessed = preprocess_input(image * 255)
        
        # Extract features
        features = self.model.predict(image_preprocessed, verbose=0)
        
        return features.flatten()


class MultiModalFeatureExtractor:
    """
    Combined feature extractor for multimodal images
    """
    def __init__(self):
        self.dwt_extractor = DWTFeatureExtractor()
        self.lbp_extractor = LBPFeatureExtractor()
        self.densenet_extractor = DenseNetFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, images, modality='mri'):
        """
        Extract features based on modality
        modality: 'mri', 'pet', or 'ct'
        """
        features_list = []
        
        for img_data in images:
            # Handle different image shapes - ensure we have a proper 3D image
            if len(img_data.shape) == 4:  # (2, H, W, C) - take first image
                img = img_data[0]
            else:
                img = img_data
                
            if modality == 'mri':
                features = self.dwt_extractor.extract(img)
            elif modality == 'pet':
                features = self.lbp_extractor.extract(img)
            elif modality == 'ct':
                features = self.densenet_extractor.extract(img)
            else:
                raise ValueError(f"Unknown modality: {modality}")
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def extract_all_modalities(self, images):
        """
        Extract features from all modalities and concatenate
        For this project, we'll treat the same brain images as all modalities
        """
        print("Extracting DWT features (MRI)...")
        dwt_features = self.extract_features(images, modality='mri')
        
        print("Extracting LBP features (PET)...")
        lbp_features = self.extract_features(images, modality='pet')
        
        print("Extracting DenseNet features (CT)...")
        densenet_features = self.extract_features(images, modality='ct')
        
        # Find minimum length for padding/truncation
        min_len = min(dwt_features.shape[1], lbp_features.shape[1], densenet_features.shape[1])
        
        # Truncate to same length
        dwt_features = dwt_features[:, :min_len*5]  # DWT gets more weight
        lbp_features = np.tile(lbp_features, (1, min_len // lbp_features.shape[1] + 1))[:, :min_len]
        densenet_features = densenet_features[:, :min_len]
        
        # Concatenate all features
        combined_features = np.concatenate([dwt_features, lbp_features, densenet_features], axis=1)
        
        return combined_features
    
    def fit_scaler(self, features):
        """Fit the feature scaler"""
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform(self, features):
        """Normalize features"""
        if not self.is_fitted:
            self.fit_scaler(features)
        return self.scaler.transform(features)


if __name__ == "__main__":
    print("Feature extraction module loaded successfully!")
    print(f"DWT Wavelet: {config.DWT_WAVELET}, Level: {config.DWT_LEVEL}")
    print(f"LBP Radius: {config.LBP_RADIUS}, Points: {config.LBP_POINTS}")
    print("DenseNet121 ready for feature extraction")
