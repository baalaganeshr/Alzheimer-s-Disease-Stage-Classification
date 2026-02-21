"""
Main Training Script for Alzheimer's Disease Stage Classification
Implements the complete pipeline with multiple learning rates
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import config
from preprocessing import ImagePreprocessor, preprocess_dataset
from feature_extraction import MultiModalFeatureExtractor
from feature_fusion import MultiModalFusionPipeline
from model import AlzheimerClassifier
from utils import (plot_training_history, plot_confusion_matrix, plot_roc_curves,
                  print_classification_metrics, save_preprocessing_steps,
                  plot_learning_rate_comparison)
from report_generator import generate_all_reports
import warnings
import pickle
warnings.filterwarnings('ignore')

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  
        print(f"[INFO] GPU available: {len(gpus)} device(s) - GTX 3050 ready for training")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("[WARNING] No GPU detected - using CPU (training will be slower)")


def load_dataset():
    """
    Load dataset from directory structure
    """
    print("Loading dataset...")
    
    image_paths = []
    labels = []
    label_map = {class_name: idx for idx, class_name in enumerate(config.CLASSES)}
    
    for class_name in config.CLASSES:
        class_path = os.path.join(config.DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory not found: {class_path}")
            continue
        
        class_images = glob.glob(os.path.join(class_path, '*.jpg'))
        class_images.extend(glob.glob(os.path.join(class_path, '*.png')))
        
        image_paths.extend(class_images)
        labels.extend([label_map[class_name]] * len(class_images))
        
        print(f"  {class_name}: {len(class_images)} images")
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return image_paths, np.array(labels)


def split_dataset(image_paths, labels):
    """
    Split dataset into train, validation, and test sets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_SEED
    )
    
    # Second split: separate train and validation
    val_size = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=config.RANDOM_SEED
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """
    Main training pipeline
    """
    print("="*80)
    print("ALZHEIMER'S DISEASE STAGE CLASSIFICATION")
    print("Multimodal Framework: DWT + LBP + DenseNet")
    print("="*80)
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    
    # Cache file path
    cache_file = os.path.join(config.OUTPUT_DIR, 'features_cache.pkl')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print("\n[INFO] Loading cached features...")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            X_train_fused = cache_data['X_train_fused']
            X_val_fused = cache_data['X_val_fused']
            X_test_fused = cache_data['X_test_fused']
            y_train = cache_data['y_train']
            y_val = cache_data['y_val']
            y_test = cache_data['y_test']
            print(f"[INFO] Cached features loaded successfully!")
            print(f"  Train: {X_train_fused.shape}, Val: {X_val_fused.shape}, Test: {X_test_fused.shape}")
            skip_preprocessing = True
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}")
            print("[INFO] Will reprocess from scratch...")
            skip_preprocessing = False
    else:
        skip_preprocessing = False
    
    if not skip_preprocessing:
        # Step 1: Load dataset
        print("\n[Step 1/7] Loading Dataset...")
        image_paths, labels = load_dataset()
        
        # Step 2: Split dataset
        print("\n[Step 2/7] Splitting Dataset...")
        X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = split_dataset(
            image_paths, labels
        )
        
        # Step 3: Preprocessing
        print("\n[Step 3/7] Preprocessing Images...")
        preprocessor = ImagePreprocessor()
        
        # Save preprocessing steps for a few sample images
        print("  Saving preprocessing visualization...")
        for idx in range(min(3, len(X_train_paths))):
            import cv2
            img = cv2.imread(X_train_paths[idx])
            _, steps = preprocessor.preprocess(img, save_steps=True)
            class_name = config.CLASSES[y_train[idx]]
            save_preprocessing_steps(steps, class_name, idx)
        
        # Preprocess all images
        print("  Preprocessing training set...")
        X_train, y_train = preprocess_dataset(X_train_paths, y_train, preprocessor)
        
        print("  Preprocessing validation set...")
        X_val, y_val = preprocess_dataset(X_val_paths, y_val, preprocessor)
        
        print("  Preprocessing test set...")
        X_test, y_test = preprocess_dataset(X_test_paths, y_test, preprocessor)
        
        print(f"  Preprocessed shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # Step 4: Feature Extraction
        print("\n[Step 4/7] Extracting Multimodal Features...")
        feature_extractor = MultiModalFeatureExtractor()
        
        print("  Extracting features from training set...")
        X_train_features = feature_extractor.extract_all_modalities(X_train)
        
        print("  Extracting features from validation set...")
        X_val_features = feature_extractor.extract_all_modalities(X_val)
        
        print("  Extracting features from test set...")
        X_test_features = feature_extractor.extract_all_modalities(X_test)
        
        print(f"  Feature shapes: Train={X_train_features.shape}, Val={X_val_features.shape}, Test={X_test_features.shape}")
        
        # Step 5: Feature Fusion and PCA
        print("\n[Step 5/7] Applying Feature Fusion with PCA...")
        fusion_pipeline = MultiModalFusionPipeline(n_components=config.PCA_COMPONENTS)
        
        print("  Fitting PCA on training features...")
        X_train_fused = fusion_pipeline.fit_transform(X_train_features)
        
        print("  Transforming validation and test features...")
        X_val_fused = fusion_pipeline.transform(X_val_features)
        X_test_fused = fusion_pipeline.transform(X_test_features)
        
        print(f"  Fused feature shapes: {X_train_fused.shape}")
        
        # Save fusion pipeline
        fusion_pipeline.save()
        
        # Save cache
        print("\n[INFO] Saving features to cache...")
        cache_data = {
            'X_train_fused': X_train_fused,
            'X_val_fused': X_val_fused,
            'X_test_fused': X_test_fused,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[INFO] Cache saved to {cache_file}")
    
    # Step 6: Train with Multiple Learning Rates
    print("\n[Step 6/7] Training Models with Multiple Learning Rates...")
    
    all_histories = {}
    all_metrics = {}
    all_predictions = {}
    
    for lr in config.LEARNING_RATES:
        print(f"\n{'='*70}")
        print(f"Training with Learning Rate: {lr}")
        print(f"{'='*70}")
        
        # Create model
        classifier = AlzheimerClassifier(
            input_dim=X_train_fused.shape[1],
            learning_rate=lr,
            num_classes=config.NUM_CLASSES
        )
        
        # Train model
        history = classifier.train(
            X_train_fused, y_train,
            X_val_fused, y_val,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            model_name=f'model_lr{lr}'
        )
        
        # Evaluate on test set
        print(f"\nEvaluating model (LR={lr}) on test set...")
        metrics = classifier.evaluate(X_test_fused, y_test)
        
        # Get predictions
        y_pred_proba = classifier.predict(X_test_fused)
        y_pred = classifier.predict_classes(X_test_fused)
        
        # Store results
        all_histories[lr] = history
        all_metrics[lr] = metrics
        all_predictions[lr] = (y_pred, y_pred_proba)
        
        # Save model
        classifier.save()
        
        print(f"\nModel (LR={lr}) training completed!")
    
    # Step 7: Generate Visualizations and Metrics
    print("\n[Step 7/7] Generating Visualizations and Reports...")
    
    for lr in config.LEARNING_RATES:
        print(f"\n  Processing results for LR={lr}...")
        
        y_pred, y_pred_proba = all_predictions[lr]
        
        # Plot training curves
        plot_training_history(all_histories[lr], lr)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, lr)
        
        # Plot ROC curves
        plot_roc_curves(y_test, y_pred_proba, lr)
        
        # Print classification metrics
        print(f"\n  Detailed Metrics for LR={lr}:")
        print_classification_metrics(y_test, y_pred, y_pred_proba)
    
    # Compare learning rates
    print("\n  Creating learning rate comparison plots...")
    plot_learning_rate_comparison(all_histories, all_metrics)
    
    # Generate structured reports
    generate_all_reports(all_metrics, all_histories, all_predictions, y_test)
    
    # Console summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    # Quick summary to console
    print("\nQUICK RESULTS:")
    for lr in config.LEARNING_RATES:
        metrics = all_metrics[lr]
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
        elif 'compile_metrics' in metrics:
            acc = metrics['compile_metrics']
        else:
            acc = list(metrics.values())[1] if len(metrics.values()) > 1 else 0.0
        print(f"  LR {lr}: Accuracy = {acc*100:.2f}%, Loss = {metrics['loss']:.4f}")
    
    print(f"\n📁 All outputs saved to: {config.OUTPUT_DIR}")
    print(f"📁 All models saved to: {config.MODEL_DIR}")
    print("\n✅ Check the generated report files for detailed results!")
    print("="*80)


if __name__ == "__main__":
    main()
