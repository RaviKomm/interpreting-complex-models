"""
evaluate.py
Computes evaluation metrics and saves a short text report and ROC curve.
"""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

    report_path = os.path.join(FIG_DIR, 'eval_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'accuracy: {acc:.4f}\n')
        f.write(f'precision: {prec:.4f}\n')
        f.write(f'recall: {rec:.4f}\n')
        f.write(f'f1: {f1:.4f}\n')
        f.write(f'roc_auc: {auc:.4f}\n')
    print(f'Saved evaluation report to {report_path}')

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.grid(True)
        path = os.path.join(FIG_DIR, 'roc_curve.png')
        plt.savefig(path, dpi=200)
        plt.close()
        print(f'Saved ROC curve to {path}')

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}
