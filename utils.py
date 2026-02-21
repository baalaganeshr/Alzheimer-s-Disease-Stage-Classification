"""
Utility functions for visualization and metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, f1_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import config
import os
import cv2


def plot_training_history(history, learning_rate, save_dir=config.OUTPUT_DIR):
    """
    Plot training accuracy and loss curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'Model Accuracy (LR={learning_rate})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'Model Loss (LR={learning_rate})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'training_curves_lr{learning_rate}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, learning_rate, save_dir=config.OUTPUT_DIR):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, yticklabels=config.CLASSES,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    
    plt.title(f'Confusion Matrix (LR={learning_rate})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'confusion_matrix_lr{learning_rate}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(y_true, y_pred_proba, learning_rate, save_dir=config.OUTPUT_DIR):
    """
    Plot ROC curves for all classes
    """
    n_classes = len(config.CLASSES)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(n_classes):
        if n_classes == 2 and i == 1:
            # For binary classification, only plot one curve
            fpr[i], tpr[i], _ = roc_curve(y_true_bin.ravel(), y_pred_proba[:, 1])
        else:
            if n_classes > 2:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            else:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_pred_proba[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                label=f'{config.CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title(f'ROC Curves (LR={learning_rate})', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'roc_curves_lr{learning_rate}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {save_path}")
    
    return roc_auc


def print_classification_metrics(y_true, y_pred, y_pred_proba):
    """
    Print detailed classification metrics
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4))
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nWeighted F1 Score: {f1:.4f}")
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("="*70)


def save_preprocessing_steps(steps, class_name, img_idx, save_dir=config.PROCESSED_DIR):
    """
    Save preprocessing step images
    """
    class_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(steps), figsize=(20, 4))
    
    if len(steps) == 1:
        axes = [axes]
    
    for idx, (step_name, img) in enumerate(steps.items()):
        if len(img.shape) == 2:
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(step_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(class_dir, f'preprocessing_steps_{img_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_rate_comparison(all_histories, all_metrics, save_dir=config.OUTPUT_DIR):
    """
    Compare results across different learning rates
    """
    learning_rates = list(all_histories.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Final Accuracy Comparison
    # Handle different possible metric key names
    final_accuracies = []
    for lr in learning_rates:
        if 'accuracy' in all_metrics[lr]:
            final_accuracies.append(all_metrics[lr]['accuracy'])
        elif 'acc' in all_metrics[lr]:
            final_accuracies.append(all_metrics[lr]['acc'])
        else:
            # If neither key exists, try to get the second value (assuming [loss, accuracy])
            metric_values = list(all_metrics[lr].values())
            final_accuracies.append(metric_values[1] if len(metric_values) > 1 else 0.0)
    axes[0, 0].bar(range(len(learning_rates)), final_accuracies, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xticks(range(len(learning_rates)))
    axes[0, 0].set_xticklabels([f'{lr}' for lr in learning_rates])
    axes[0, 0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Test Accuracy vs Learning Rate', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Training Accuracy Curves
    for lr in learning_rates:
        axes[0, 1].plot(all_histories[lr].history['accuracy'], label=f'LR={lr}', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy Curves
    for lr in learning_rates:
        axes[1, 0].plot(all_histories[lr].history['val_accuracy'], label=f'LR={lr}', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Curves
    for lr in learning_rates:
        axes[1, 1].plot(all_histories[lr].history['loss'], label=f'LR={lr}', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'learning_rate_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning rate comparison saved to {save_path}")


if __name__ == "__main__":
    print("Utils module loaded successfully!")
