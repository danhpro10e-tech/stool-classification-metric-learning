import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class StoolMetrics:
    """Theo dõi các metrics quan trọng cho phân loại ảnh phân"""
    
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, preds, targets, loss=None):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        if loss is not None:
            self.losses.append(loss)
            
    def compute(self):
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall accuracy
        accuracy = accuracy_score(targets, preds)
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(self.num_classes):
            mask = targets == i
            if mask.sum() > 0:
                acc = (preds[mask] == targets[mask]).sum() / mask.sum()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        # Mean per class accuracy (MPCA)
        mpca = np.mean(per_class_acc)
        
        # Mean Absolute Deviation (MAD) - theo thang BSS
        abs_deviation = np.abs(preds - targets)
        mad_overall = np.mean(abs_deviation)
        mad_per_class = [np.mean(abs_deviation[targets == i]) if (targets == i).sum() > 0 else 0.0 
                         for i in range(self.num_classes)]
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        # Mean loss
        mean_loss = np.mean(self.losses) if self.losses else 0.0
        
        return {
            'accuracy': accuracy,
            'mpca': mpca,
            'per_class_accuracy': per_class_acc,
            'mad_overall': mad_overall,
            'mad_per_class': mad_per_class,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_loss': mean_loss
        }
    
    def print_summary(self):
        metrics = self.compute()
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Mean Per Class Accuracy: {metrics['mpca']*100:.2f}%")
        print(f"Mean Absolute Deviation (BSS): {metrics['mad_overall']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        
        print("\nPer-Class Performance:")
        print("-" * 40)
        for i in range(self.num_classes):
            print(f"Class {i}: Acc={metrics['per_class_accuracy'][i]*100:.1f}%, "
                  f"MAD={metrics['mad_per_class'][i]:.3f}")
        print("="*50)
        return metrics