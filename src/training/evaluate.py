import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for audio, targets, filenames in test_loader:
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(audio)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions),
            'recall': recall_score(all_targets, all_predictions),
            'f1_score': f1_score(all_targets, all_predictions),
            'auc': roc_auc_score(all_targets, all_probabilities)
        }
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_targets, all_predictions)
        
        # Plot ROC curve
        self._plot_roc_curve(all_targets, all_probabilities)
        
        return metrics
    
    def _plot_confusion_matrix(self, targets: List[int], predictions: List[int]):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Bonafide', 'Spoof'],
                   yticklabels=['Bonafide', 'Spoof'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self, targets: List[int], probabilities: List[float]):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
