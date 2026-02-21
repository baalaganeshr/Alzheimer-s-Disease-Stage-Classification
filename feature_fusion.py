"""
Feature Fusion and Dimensionality Reduction using PCA
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config
import joblib
import os


class FeatureFusion:
    """
    Feature-level fusion with PCA for dimensionality reduction
    """
    def __init__(self, n_components=config.PCA_COMPONENTS):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features):
        """
        Fit PCA on features
        """
        # Normalize features first
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit PCA
        self.pca.fit(features_scaled)
        self.is_fitted = True
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self
    
    def transform(self, features):
        """
        Transform features using fitted PCA
        """
        if not self.is_fitted:
            raise ValueError("FeatureFusion must be fitted before transform")
        
        # Normalize and apply PCA
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        return features_pca
    
    def fit_transform(self, features):
        """
        Fit and transform in one step
        """
        self.fit(features)
        return self.transform(features)
    
    def save(self, filepath=None):
        """
        Save the fusion model
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, 'feature_fusion.pkl')
        
        joblib.dump({
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }, filepath)
        
        print(f"Feature fusion model saved to {filepath}")
    
    def load(self, filepath=None):
        """
        Load the fusion model
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, 'feature_fusion.pkl')
        
        data = joblib.load(filepath)
        self.pca = data['pca']
        self.scaler = data['scaler']
        self.n_components = data['n_components']
        self.is_fitted = data['is_fitted']
        
        print(f"Feature fusion model loaded from {filepath}")
        return self
    
    def get_explained_variance(self):
        """
        Get explained variance ratio
        """
        if not self.is_fitted:
            return None
        return self.pca.explained_variance_ratio_


def fuse_multimodal_features(mri_features, pet_features, ct_features, 
                             fusion_strategy='concatenate'):
    """
    Fuse features from multiple modalities
    
    Args:
        mri_features: DWT features from MRI
        pet_features: LBP features from PET
        ct_features: DenseNet features from CT
        fusion_strategy: 'concatenate' or 'average'
    
    Returns:
        Fused features
    """
    if fusion_strategy == 'concatenate':
        # Simple concatenation
        fused = np.concatenate([mri_features, pet_features, ct_features], axis=1)
    
    elif fusion_strategy == 'average':
        # Average pooling (requires same dimensions)
        # Pad shorter features to match longest
        max_len = max(mri_features.shape[1], pet_features.shape[1], ct_features.shape[1])
        
        mri_padded = np.pad(mri_features, ((0, 0), (0, max_len - mri_features.shape[1])))
        pet_padded = np.pad(pet_features, ((0, 0), (0, max_len - pet_features.shape[1])))
        ct_padded = np.pad(ct_features, ((0, 0), (0, max_len - ct_features.shape[1])))
        
        fused = (mri_padded + pet_padded + ct_padded) / 3
    
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    return fused


class MultiModalFusionPipeline:
    """
    Complete pipeline for multimodal feature extraction and fusion
    """
    def __init__(self, n_components=config.PCA_COMPONENTS):
        self.fusion = FeatureFusion(n_components=n_components)
    
    def fit(self, combined_features):
        """
        Fit the fusion pipeline
        """
        self.fusion.fit(combined_features)
        return self
    
    def transform(self, combined_features):
        """
        Transform features through the pipeline
        """
        return self.fusion.transform(combined_features)
    
    def fit_transform(self, combined_features):
        """
        Fit and transform
        """
        return self.fusion.fit_transform(combined_features)
    
    def save(self, filepath=None):
        """
        Save the pipeline
        """
        self.fusion.save(filepath)
    
    def load(self, filepath=None):
        """
        Load the pipeline
        """
        self.fusion.load(filepath)
        return self


if __name__ == "__main__":
    print("Feature fusion module ready!")
    print(f"PCA components: {config.PCA_COMPONENTS}")
