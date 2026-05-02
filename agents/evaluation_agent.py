import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from config import RESULTS_DIR

class EvaluationAgent:
    """
    Agent responsible for evaluating model performance.
    Single Responsibility: Calculate and export metrics.
    """
    def __init__(self):
        self.results_dir = RESULTS_DIR

    def evaluate(self, predictions):
        """
        Calculates Top-1, Top-5 accuracy, plots Confusion Matrix,
        and saves results to disk.
        """
        if predictions is None:
            print("[EvaluationAgent] No predictions to evaluate.")
            return

        y_true = predictions['y_true']
        y_pred_top1 = predictions['y_pred_top1']
        y_pred_top5 = predictions['y_pred_top5']

        # Calculate Top-1 Accuracy
        top1_acc = accuracy_score(y_true, y_pred_top1)
        
        # Calculate Top-5 Accuracy
        top5_correct = 0
        for true_lbl, top5 in zip(y_true, y_pred_top5):
            if true_lbl in top5:
                top5_correct += 1
        top5_acc = top5_correct / len(y_true) if len(y_true) > 0 else 0.0

        metrics = {
            "top1_accuracy": float(top1_acc),
            "top5_accuracy": float(top5_acc),
            "total_test_samples": len(y_true)
        }

        # Save metrics to JSON
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"[EvaluationAgent] Evaluation Results:")
        print(f" - Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
        print(f" - Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")
        print(f" - Results saved to {metrics_path}")

        # Generate and save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_top1)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix (Top-1)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        print(f"[EvaluationAgent] Confusion matrix saved to {cm_path}")
