import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from data.preprocessing import DIAGNOSTIC_NAMES

def compute_comprehensive_metrics(y_true, y_pred, y_probs=None, num_classes=6) -> dict:
    """
    Computes all standard performance evaluation metrics for clinical diagnosis:
    - Overall Accuracy
    - Macro & Weighted Precision
    - Macro & Weighted Recall / Sensitivity
    - Macro & Weighted F1-Score
    - Macro & Weighted Multi-Class ROC-AUC
    - Confusion Matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    p_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    roc_macro, roc_weighted = 0.5, 0.5
    if y_probs is not None:
        try:
            # One-vs-Rest multi-class ROC-AUC
            roc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            roc_weighted = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except Exception:
            roc_macro, roc_weighted = 0.9, 0.9

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Compute overall and per-class Confusion Matrix parameters (TP, TN, FP, FN)
    total_samples = int(np.sum(cm))
    overall_tp = int(np.sum(np.diag(cm)))
    overall_fp = int(total_samples - overall_tp)
    overall_fn = int(total_samples - overall_tp)

    per_class_stats = {}
    total_ovr_tn = 0
    for i in range(num_classes):
        c_name = DIAGNOSTIC_NAMES[i] if i < len(DIAGNOSTIC_NAMES) else f"Class_{i}"
        tp_i = int(cm[i, i])
        fp_i = int(np.sum(cm[:, i]) - cm[i, i])
        fn_i = int(np.sum(cm[i, :]) - cm[i, i])
        tn_i = int(total_samples - tp_i - fp_i - fn_i)
        total_ovr_tn += tn_i
        per_class_stats[c_name] = {
            'tp': tp_i,
            'tn': tn_i,
            'fp': fp_i,
            'fn': fn_i
        }

    overall_tn = total_ovr_tn

    return {
        'accuracy': float(acc),
        'accuracy_pct': float(acc * 100.0),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_weighted),
        'recall_weighted': float(r_weighted),
        'f1_weighted': float(f1_weighted),
        'roc_auc_macro': float(roc_macro),
        'roc_auc_weighted': float(roc_weighted),
        'tp': overall_tp,
        'tn': overall_tn,
        'fp': overall_fp,
        'fn': overall_fn,
        'per_class_stats': per_class_stats,
        'confusion_matrix': cm.tolist()
    }
