"""
Simplified Alzheimer's Classification Training
Avoids TensorFlow 2.20/Keras 3.13 compatibility issues
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models, layers, callbacks
import tensorflow as tf

# Disable GPU if causing issues
# tf.config.set_visible_devices([], 'GPU')

print("=" * 80)
print("ALZHEIMER'S DISEASE STAGE CLASSIFICATION")
print("Simplified Training Pipeline")
print("=" * 80)

# Configuration
IMAGE_SIZE = 128  # Reduced for faster training
DATA_DIR = Path(r"c:\Users\baala\OneDrive\Desktop\mlc\dataset\Data")
OUTPUT_DIR = Path("./outputs")
MODELS_DIR = Path("./models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
EPOCHS = 15  # Reduced for faster completion
BATCH_SIZE = 32

# Load Dataset
print("\n[1/5] Loading Dataset...", flush=True)
images = []
labels = []
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

for class_idx, class_name in enumerate(class_names):
    class_dir = DATA_DIR / class_name
    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    print(f"  Loading {class_name}: {len(image_files)} images", flush=True)
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(class_idx)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\nTotal samples: {len(X)}")
print(f"Classes: {class_names}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Data shape: {X.shape}")

# Split Dataset
print("\n[2/5] Splitting Dataset...", flush=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Training Loop
all_histories = {}
all_models = {}
best_model = None
best_acc = 0
best_lr = None

print("\n[3/5] Training Models with Different Learning Rates...", flush=True)

for lr in LEARNING_RATES:
    print(f"\n  Training with LR = {lr}...", flush=True)
    
    # Build CNN model (not fully connected to avoid memory issues)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Custom callback to print progress
    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"    Epoch {epoch+1}/{EPOCHS} - "
                  f"loss: {logs['loss']:.4f} - acc: {logs['accuracy']:.4f} - "
                  f"val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_accuracy']:.4f}", 
                  flush=True)
    
    # Train using explicit numpy arrays
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[ProgressCallback()]
    )
    
    all_histories[lr] = history
    all_models[lr] = model
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"    Final validation accuracy: {final_acc:.4f}", flush=True)
    
    if final_acc > best_acc:
        best_acc = final_acc
        best_model = model
        best_lr = lr
    
    # Save model
    model.save(MODELS_DIR / f"model_lr_{lr}.keras")

print(f"\n  Best model: LR={best_lr} with val_acc={best_acc:.4f}")

# Evaluate on test set
print("\n[4/5] Evaluating Best Model...", flush=True)
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification Report
report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(f"Best Model: LR={best_lr}, Val Accuracy={best_acc:.4f}\n\n")
    f.write(report)

# Generate Visualizations
print("\n[5/5] Generating Visualizations...", flush=True)

# 1. Accuracy Curves
plt.figure(figsize=(12, 8))
for lr in LEARNING_RATES:
    history = all_histories[lr]
    plt.plot(history.history['val_accuracy'], label=f'LR={lr}', linewidth=2)
plt.title('Validation Accuracy vs Epochs for Different Learning Rates', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Accuracy curves saved")

# 2. Loss Curves
plt.figure(figsize=(12, 8))
for lr in LEARNING_RATES:
    history = all_histories[lr]
    plt.plot(history.history['loss'], label=f'LR={lr} (train)', linewidth=2)
    plt.plot(history.history['val_loss'], label=f'LR={lr} (val)', linewidth=2, linestyle='--')
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Loss curves saved")

# 3. Confusion Matrices (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, lr in enumerate(LEARNING_RATES):
    model = all_models[lr]
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f'LR={lr}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices for Different Learning Rates', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Confusion matrices saved")

# 4. ROC Curves
plt.figure(figsize=(10, 8))
for class_idx, class_name in enumerate(class_names):
    y_true_binary = (y_test == class_idx).astype(int)
    y_score = y_pred_proba[:, class_idx]
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves - Best Model (LR={best_lr})', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ ROC curves saved")

# 5. Learning Rate Comparison
final_accs = [all_histories[lr].history['val_accuracy'][-1] for lr in LEARNING_RATES]
plt.figure(figsize=(10, 6))
bars = plt.bar([str(lr) for lr in LEARNING_RATES], final_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
plt.title('Final Validation Accuracy by Learning Rate', fontsize=14, fontweight='bold')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, final_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_rate_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Learning rate comparison saved")

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nBest Model: LR={best_lr}, Validation Accuracy={best_acc:.4f}")
print(f"Test Accuracy: {np.mean(y_pred == y_test):.4f}")
print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
print(f"Models saved to: {MODELS_DIR.absolute()}")
print("\nGenerated files:")
print("  - accuracy_curves.png")
print("  - loss_curves.png")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - learning_rate_comparison.png")
print("  - classification_report.txt")
