from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate all classification metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=labels
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics

def print_metrics(metrics, model_name="Model"):
    """
    Pretty print metrics
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*50}\n")

def save_metrics_to_csv(metrics_dict, filename, results_dir='../results'):
    """
    Save metrics comparison to CSV
    """
    import os
    df = pd.DataFrame(metrics_dict).T
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath)
    print(f"Metrics saved to {filepath}")