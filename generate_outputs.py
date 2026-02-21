"""
Generate Visualizations from Saved Models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path
import pickle

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS FROM TRAINING RESULTS")
print("=" * 80)

# Paths
OUTPUT_DIR = Path("./outputs")
MODELS_DIR = Path("./models")

# Load saved data
print("\n[1/3] Loading saved training results...")
with open(MODELS_DIR / 'training_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']
class_names = data['class_names']
all_histories = data['all_histories']
best_lr = data['best_lr']
all_predictions = data['all_predictions']

LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]

print(f"  Loaded test data: {X_test.shape}")
print(f"  Best LR: {best_lr}")

# Generate Visualizations
print("\n[2/3] Generating Visualizations...")

# 1. Accuracy Curves
plt.figure(figsize=(14, 6))
for lr in LEARNING_RATES:
    history = all_histories[lr]
    plt.plot(history['val_accuracy'], label=f'LR={lr}', linewidth=2.5)
plt.title('Validation Accuracy vs Epochs for Different Learning Rates', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Validation Accuracy', fontsize=13)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Accuracy curves saved")

# 2. Loss Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Training loss
for lr in LEARNING_RATES:
    history = all_histories[lr]
    ax1.plot(history['loss'], label=f'LR={lr}', linewidth=2.5)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Validation loss  
for lr in LEARNING_RATES:
    history = all_histories[lr]
    ax2.plot(history['val_loss'], label=f'LR={lr}', linewidth=2.5)
ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Model Loss Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Loss curves saved")

# 3. Confusion Matrices (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.flatten()

for idx, lr in enumerate(LEARNING_RATES):
    y_pred = all_predictions[lr]
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    
    # Calculate accuracy for this LR
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    axes[idx].set_title(f'LR={lr} (Accuracy: {acc:.2%})', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_ylabel('Actual', fontsize=11)

plt.suptitle('Confusion Matrices for Different Learning Rates', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Confusion matrices saved")

# 4. ROC Curves (for best model)
plt.figure(figsize=(10, 8))
best_pred_proba = data['best_pred_proba']

colors = ['#FF6B6B', '#4ECDC4']
for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
    y_true_binary = (y_test == class_idx).astype(int)
    y_score = best_pred_proba[:, class_idx]
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', 
             linewidth=3, color=color)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title(f'ROC Curves - Best Model (LR={best_lr})', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] ROC curves saved")

# 5. Learning Rate Comparison
final_accs = [all_histories[lr]['val_accuracy'][-1] for lr in LEARNING_RATES]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

plt.figure(figsize=(12, 7))
bars = plt.bar([str(lr) for lr in LEARNING_RATES], final_accs, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Final Validation Accuracy by Learning Rate', fontsize=16, fontweight='bold')
plt.xlabel('Learning Rate', fontsize=13)
plt.ylabel('Validation Accuracy', fontsize=13)
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc, lr in zip(bars, final_accs, LEARNING_RATES):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
             f'{acc:.4f}\\n({acc*100:.2f}%)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Mark best model
    if lr == best_lr:
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.09,
                 '★ BEST ★', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold', color='gold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='darkblue', alpha=0.8, edgecolor='gold', linewidth=2))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Learning rate comparison saved")

# 6. Classification Report  
print("\n[3/3] Generating Classification Report...")
y_pred_best = all_predictions[best_lr]
report = classification_report(y_test, y_pred_best, target_names=class_names)

with open(OUTPUT_DIR / 'classification_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ALZHEIMER'S DISEASE CLASSIFICATION - FINAL RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Best Model: Learning Rate = {best_lr}\n")
    f.write(f"Best Validation Accuracy: {all_histories[best_lr]['val_accuracy'][-1]:.4f}\n\n")
    f.write("Learning Rate Comparison:\n")
    f.write("-" * 40 + "\n")
    for lr in LEARNING_RATES:
        acc = all_histories[lr]['val_accuracy'][-1]
        marker = " <- BEST" if lr == best_lr else ""
        f.write(f"  LR {lr:>7}: {acc:.4f} ({acc*100:.2f}%){marker}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("CLASSIFICATION REPORT (TEST SET)\n")
    f.write("=" * 80 + "\n\n")
    f.write(report)
    f.write("\n" + "=" * 80 + "\n")

print("  [OK] Classification report saved")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
print(f"\nOutput Directory: {OUTPUT_DIR.absolute()}")
print("\nGenerated Files:")
print("  1. accuracy_curves.png")
print("  2. loss_curves.png")
print("  3. confusion_matrices.png")
print("  4. roc_curves.png")
print("  5. learning_rate_comparison.png")
print("  6. classification_report.txt")
print("\n" + "=" * 80)
