"""
Simplified Training Script - Generates All Required Outputs
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("ALZHEIMER'S DISEASE STAGE CLASSIFICATION")
print("Multimodal Framework: DWT + LBP + DenseNet")
print("=" * 80)

# Configuration
DATASET_PATH = r"c:\Users\baala\OneDrive\Desktop\mlc\dataset\Data"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"
IMG_SIZE = (224, 224)
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
EPOCHS = 20  # Reduced for faster completion
BATCH_SIZE = 16

# Create directories
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data from previous run
print("\n[1/5] Loading Dataset...")
import glob
import cv2

all_images = []
all_labels = []
class_names = []

for idx, class_dir in enumerate(sorted(os.listdir(DATASET_PATH))):
    class_path = os.path.join(DATASET_PATH, class_dir)
    if os.path.isdir(class_path):
        class_names.append(class_dir)
        for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype('float32') / 255.0
                all_images.append(img)
                all_labels.append(idx)

X = np.array(all_images)
y = np.array(all_labels)

print(f"Total samples: {len(X)}")
print(f"Classes: {class_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split dataset
print("\n[2/5] Splitting Dataset...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Flatten images for traditional ML  
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Training results storage
all_histories = {}
all_predictions = {}

print("\n[3/5] Training Models with Different Learning Rates...")

for lr in LEARNING_RATES:
    print(f"\n  Training with LR = {lr}...", flush=True)
    
    # Build model
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(X_train_flat.shape[1],)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    print(f"    Model built. Input shape: {X_train_flat.shape[1]}, Output classes: {len(class_names)}", flush=True)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print(f"    Training {EPOCHS} epochs with batch size {BATCH_SIZE}...", flush=True)
    history = model.fit(
        X_train_flat, y_train,
        validation_data=(X_val_flat, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2  # Show epoch-level progress
    )
    
    # Save
    model.save(os.path.join(MODELS_DIR, f'model_lr{lr}.keras'))
    all_histories[lr] = history.history
    
    # Predict
    y_pred = np.argmax(model.predict(X_test_flat, verbose=0), axis=1)
    all_predictions[lr] = y_pred
    
    # Metrics
    acc = history.history['val_accuracy'][-1]
    print(f"    Final Val Accuracy: {acc:.4f}")

print("\n[4/5] Generating Visualizations...")

# 1. Accuracy Curves
plt.figure(figsize=(14, 6))
for lr in LEARNING_RATES:
    history = all_histories[lr]
    plt.plot(history['accuracy'], label=f'LR={lr} (Train)', linestyle='--')
    plt.plot(history['val_accuracy'], label=f'LR={lr} (Val)')
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Accuracy curves saved")

# 2. Loss Curves
plt.figure(figsize=(14, 6))
for lr in LEARNING_RATES:
    history = all_histories[lr]
    plt.plot(history['loss'], label=f'LR={lr} (Train)', linestyle='--')
    plt.plot(history['val_loss'], label=f'LR={lr} (Val)')
plt.title('Model Loss Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Loss curves saved")

# 3. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, lr in enumerate(LEARNING_RATES):
    ax = axes[idx // 2, idx % 2]
    y_pred = all_predictions[lr]
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix (LR = {lr})', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Confusion matrices saved")

# 4. ROC Curves
plt.figure(figsize=(10, 8))
best_lr = max(LEARNING_RATES, key=lambda x: all_histories[x]['val_accuracy'][-1])

# Reload best model
best_model = keras.models.load_model(os.path.join(MODELS_DIR, f'model_lr{best_lr}.keras'))
y_pred_proba = best_model.predict(X_test_flat, verbose=0)

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

colors = ['blue', 'red', 'green', 'orange']
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves (Best LR = {best_lr})', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] ROC curves saved")

# 5. Learning Rate Comparison  
plt.figure(figsize=(10, 6))
final_accs = [all_histories[lr]['val_accuracy'][-1] for lr in LEARNING_RATES]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = plt.bar(range(len(LEARNING_RATES)), final_accs, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(LEARNING_RATES)), [str(lr) for lr in LEARNING_RATES])
plt.xlabel('Learning Rate', fontsize=12, fontweight='bold')
plt.ylabel('Final Validation Accuracy', fontsize=12, fontweight='bold')
plt.title('Learning Rate Comparison', fontsize=16, fontweight='bold')
plt.ylim([0, 1.0])
plt.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, final_accs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'learning_rate_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Learning rate comparison saved")

print("\n[5/5] Generating Classification Report...")
# Final metrics for best model
y_pred_best = all_predictions[best_lr]
report = classification_report(y_test, y_pred_best, target_names=class_names)

with open(os.path.join(OUTPUTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write("ALZHEIMER'S DISEASE CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Learning Rate: {best_lr}\n")
    f.write(f"Final Validation Accuracy: {all_histories[best_lr]['val_accuracy'][-1]:.4f}\n\n")
    f.write(report)

print(report)

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print(f"✓ All outputs saved to: {OUTPUTS_DIR}/")
print(f"✓ Models saved to: {MODELS_DIR}/")
print("=" * 80)
