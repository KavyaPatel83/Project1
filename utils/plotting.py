import os
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_training_curves(history: dict, model_code: str, model_name: str, save_path: str):
    """
    Plots training loss and validation accuracy over epochs.
    history: dict with 'train_loss' and 'val_acc' lists
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    color = '#1f77b4'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train Loss', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(epochs, history['train_loss'], color=color, linewidth=2.2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    val_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in history['val_acc']]
    line2 = ax2.plot(epochs, val_acc_pct, color=color, linewidth=2.2, linestyle='-', marker='o', markersize=3, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combined title and legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)

    plt.title(f"{model_name} (Model {model_code})\nTraining Loss & Validation Accuracy Curves", fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=160)
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, model_code: str, model_name: str, save_path: str):
    """
    Plots annotated confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: Model {model_code}\n{model_name}",
        ylabel='True Diagnostic Label',
        xlabel='Predicted Diagnostic Label'
    )
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')

    # Annotate numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold', fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=160)
    plt.close()

def plot_roc_curves_multiclass(y_true, y_probs, classes, model_code: str, model_name: str, save_path: str):
    """
    Plots multi-class One-vs-Rest ROC curves with AUC for each diagnostic category.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in range(n_classes):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{classes[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance Line')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title(f"ROC Curves: Model {model_code} ({model_name})", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=160)
    plt.close()

def plot_model_metrics_scorecard(metrics: dict, model_code: str, model_name: str, save_path: str):
    """
    Renders an image scorecard displaying all performance scores for a single model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=160)
    ax.axis('off')

    title = f"PERFORMANCE SCORECARD: MODEL {model_code}\n{model_name}"
    ax.text(0.5, 0.94, title, ha='center', va='top', fontsize=13, fontweight='bold', color='#0b2e59')

    sens_m = metrics.get('sensitivity_macro', metrics.get('recall_macro', 0.0))
    sens_w = metrics.get('sensitivity_weighted', metrics.get('recall_weighted', 0.0))
    spec_m = metrics.get('specificity_macro', 0.0)
    spec_w = metrics.get('specificity_weighted', 0.0)
    auc_m = metrics.get('auc_macro', metrics.get('roc_auc_macro', 0.0))
    auc_w = metrics.get('auc_weighted', metrics.get('roc_auc_weighted', 0.0))
    auprc_m = metrics.get('auprc_macro', 0.0)
    auprc_w = metrics.get('auprc_weighted', 0.0)

    table_data = [
        ["Metric Parameter", "Score Value", "Clinical Benchmark Level"],
        ["Accuracy", f"{metrics.get('accuracy_pct', 0.0):.2f}%", "High Multimodal Concordance"],
        ["Sensitivity / Recall (Macro)", f"{sens_m:.4f}", "Malignancy Screening Safety"],
        ["Sensitivity / Recall (Weighted)", f"{sens_w:.4f}", "General Lesion Sensitivity"],
        ["Specificity (Macro TNR)", f"{spec_m:.4f}", "Benign Identification Specificity"],
        ["Specificity (Weighted TNR)", f"{spec_w:.4f}", "Prevalence-Adjusted TNR"],
        ["AUC (ROC-AUC Macro OvR)", f"{auc_m:.4f}", "Multi-class Discriminative Power"],
        ["AUC (ROC-AUC Weighted OvR)", f"{auc_w:.4f}", "Population-Adjusted Discrimination"],
        ["AUPRC (Precision-Recall AUC Macro)", f"{auprc_m:.4f}", "Imbalanced PR Discrimination"],
        ["AUPRC (PR-AUC Weighted)", f"{auprc_w:.4f}", "Weighted PR Performance"],
        ["Precision (Macro)", f"{metrics.get('precision_macro', 0.0):.4f}", "Balanced Cross-Class Precision"],
        ["Precision (Weighted)", f"{metrics.get('precision_weighted', 0.0):.4f}", "Prevalence-Weighted Precision"],
        ["F1-Score (Macro)", f"{metrics.get('f1_macro', 0.0):.4f}", "Harmonic Mean (Balanced)"],
        ["F1-Score (Weighted)", f"{metrics.get('f1_weighted', 0.0):.4f}", "Harmonic Mean (Weighted)"],
        ["True Positives (TP)", f"{metrics.get('tp', '-')}", "Correct Positive Diagnoses"],
        ["False Positives / Negatives (FP/FN)", f"{metrics.get('fp', '-')}", "Diagnostic Error Bound"],
        ["Dataset Size Used", "Total=2298 (Test=459)", "Stratified Test Partition (20%)"]
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.42, 0.20, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    # Style header and rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1a4971')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f2f6fa')
            else:
                cell.set_facecolor('#ffffff')
            if col == 1:
                cell.set_text_props(fontweight='bold', color='#0e3a64')

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=160)
    plt.close()

def plot_global_comparison_barchart(results_list: list, save_path: str):
    """
    Grouped bar chart comparing Accuracy, Precision, Macro F1, and Macro AUC across all 7 models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    codes = [r['code'] for r in results_list]
    accuracies = [r['accuracy_pct'] for r in results_list]
    precisions = [r['precision_macro'] * 100 for r in results_list]
    f1_macros = [r['f1_macro'] * 100 for r in results_list]
    roc_aucs = [r.get('auc_macro', r.get('roc_auc_macro', 0.0)) * 100 for r in results_list]

    x = np.arange(len(codes))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    rects1 = ax.bar(x - 1.5 * width, accuracies, width, label='Accuracy (%)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x - 0.5 * width, precisions, width, label='Precision (%)', color='#8b5cf6', edgecolor='black', alpha=0.9)
    rects3 = ax.bar(x + 0.5 * width, f1_macros, width, label='F1-Score (%)', color='#10b981', edgecolor='black', alpha=0.9)
    rects4 = ax.bar(x + 1.5 * width, roc_aucs, width, label='Macro AUC (%)', color='#f59e0b', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Performance Comparison across Models A through G\nPAD-UFES-20 Dataset (Accuracy, Precision, F1-Score, AUC)',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Model {c}\n({r['modality'].split()[0]})" for c, r in zip(codes, results_list)], fontsize=10)
    ax.legend(loc='upper left', frameon=True, fontsize=10.5)
    ax.set_ylim([0, 108])
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=160)
    plt.close()

def plot_global_scorecard_table(results_list: list, save_path: str):
    """
    Renders a master scorecard table image summarizing all metrics across all 7 models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 6.5), dpi=180)
    ax.axis('off')

    title = "DERMA-GUARD MASTER PERFORMANCE SCORECARD: MODELS A THROUGH G"
    ax.text(0.5, 0.94, title, ha='center', va='top', fontsize=14, fontweight='bold', color='#0b2e59')

    table_data = [
        ["Code", "Modality Paradigm", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score", "AUC", "AUPRC", "TP", "FP/FN", "Dataset Split"]
    ]
    for r in results_list:
        prec = f"{r.get('precision_macro', 0.0):.4f}"
        sens = f"{r.get('sensitivity_macro', r.get('recall_macro', 0.0)):.4f}"
        spec = f"{r.get('specificity_macro', 0.0):.4f}"
        f1 = f"{r.get('f1_macro', 0.0):.4f}"
        auc = f"{r.get('auc_macro', r.get('roc_auc_macro', 0.0)):.4f}"
        auprc = f"{r.get('auprc_macro', 0.0):.4f}"
        table_data.append([
            f"Model {r['code']}",
            r['modality'],
            f"{r['accuracy_pct']:.2f}%",
            prec,
            sens,
            spec,
            f1,
            auc,
            auprc,
            f"{r.get('tp', '-')}",
            f"{r.get('fp', '-')}",
            "N=2298 (1953/229/459)"
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.06, 0.21, 0.07, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.04, 0.05, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1a4971')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f0f5fa')
            else:
                cell.set_facecolor('#ffffff')
            if col in [2, 3, 4, 5, 6, 7]:
                cell.set_text_props(fontweight='bold')
            if row == len(results_list): # Model G row
                cell.set_facecolor('#e3f2fd')

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=180)
    plt.close()

def plot_benchmark_summary_table_image(benchmark_results: list, save_path: str):
    """
    Renders the entire models_benchmark_summary.csv dataset into an ultra-crisp,
    high-resolution table image containing all columns, confusion matrix parameters,
    and dataset sizes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(23, 9.2), dpi=220)
    ax.axis('off')

    title = "DERMA-GUARD MULTIMODAL COMPREHENSIVE BENCHMARK SCORECARD"
    subtitle = "Dataset: PAD-UFES-20 (Total N = 2,298 | Train N = 1,953 [85%] | Val N = 229 [10%] | Test N = 459 [20%])"
    ax.text(0.5, 0.96, title, ha='center', va='top', fontsize=15, fontweight='bold', color='#0b2e59')
    ax.text(0.5, 0.92, subtitle, ha='center', va='top', fontsize=11.5, fontstyle='italic', color='#334155')

    headers = [
        "Code", "Model Name", "Modality Paradigm",
        "Accuracy", "Precision (M)", "Sensitivity (M)", "Specificity (M)", "F1-Score (M)", "AUC (M)", "AUPRC (M)",
        "TP", "TN", "FP", "FN", "Dataset Split (Train/Val/Test)"
    ]
    table_data = [headers]

    for r in benchmark_results:
        prec = f"{r.get('precision_macro', 0.0):.4f}"
        sens = f"{r.get('sensitivity_macro', r.get('recall_macro', 0.0)):.4f}"
        spec = f"{r.get('specificity_macro', 0.0):.4f}"
        f1 = f"{r.get('f1_macro', 0.0):.4f}"
        auc = f"{r.get('auc_macro', r.get('roc_auc_macro', 0.0)):.4f}"
        auprc = f"{r.get('auprc_macro', 0.0):.4f}"
        table_data.append([
            f"Model {r['code']}",
            r['name'],
            r['modality'],
            f"{r['accuracy_pct']:.2f}%",
            prec,
            sens,
            spec,
            f1,
            auc,
            auprc,
            str(r.get('tp', '-')),
            str(r.get('tn', '-')),
            str(r.get('fp', '-')),
            str(r.get('fn', '-')),
            f"Total=2298 (1953/229/{r.get('dataset_test', 459)})"
        ])

    col_widths = [0.05, 0.14, 0.17, 0.065, 0.065, 0.065, 0.065, 0.065, 0.06, 0.06, 0.035, 0.045, 0.035, 0.035, 0.10]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.65)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f1f5f9')
            else:
                cell.set_facecolor('#ffffff')

            # Highlight text model (Model C)
            if row == 3: # Model C
                cell.set_facecolor('#fef3c7')
                if col in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
                    cell.set_text_props(fontweight='bold', color='#92400e')
            # Highlight tri-modal model (Model G)
            elif row == len(benchmark_results): # Model G
                cell.set_facecolor('#e0f2fe')
                if col in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
                    cell.set_text_props(fontweight='bold', color='#0369a1')
            else:
                if col in [3, 4, 5, 6, 7, 8, 9]:
                    cell.set_text_props(fontweight='bold', color='#0f172a')

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=220)
    plt.close()

def plot_all_models_training_curves(all_histories: dict, save_path: str):
    """
    Renders a unified, high-resolution multi-panel figure displaying training loss
    and validation accuracy curves for all 7 models in a single image.
    all_histories: dict mapping model_code -> history dict ('train_loss', 'val_acc')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    names_map = {
        'A': 'Model A (Only Image: ViT)',
        'B': 'Model B (Only Metadata: MLP)',
        'C': 'Model C (Only Free Text: ClinicalBERT)',
        'D': 'Model D (Image + Metadata: ViT + MLP)',
        'E': 'Model E (Image + Text: ViT + BERT)',
        'F': 'Model F (Text + Metadata: BERT + MLP)',
        'G': 'Model G (Full Tri-Modal: ViT + MLP + BERT)'
    }
    color_palette = {
        'A': '#2563eb',
        'B': '#7c3aed',
        'C': '#d97706',
        'D': '#059669',
        'E': '#0284c7',
        'F': '#e11d48',
        'G': '#16a34a'
    }

    fig = plt.figure(figsize=(22, 26), dpi=200)
    fig.suptitle(
        "DERMA-GUARD: TRAINING LOSS & VALIDATION ACCURACY CURVES (ALL 7 MODELS)\n"
        "Unified Multi-Modality Learning Progression Across Cutaneous Lesion Classifiers (PAD-UFES-20)",
        fontsize=18, fontweight='bold', color='#0f172a', y=0.985
    )

    # 4 rows x 2 columns grid
    # Subplots 1 to 7: Individual model dual-axis curves
    # Subplot 8: Comparative Validation Accuracy curves of all 7 models overlay
    for idx, code in enumerate(model_order):
        ax1 = fig.add_subplot(4, 2, idx + 1)
        history = all_histories.get(code, {'train_loss': [0.5], 'val_acc': [0.9]})
        epochs = range(1, len(history['train_loss']) + 1)

        # Left axis: Train Loss
        c_loss = '#dc2626'
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Training Loss', color=c_loss, fontsize=11, fontweight='bold')
        l1 = ax1.plot(epochs, history['train_loss'], color=c_loss, linewidth=2.4, linestyle='-', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=c_loss)
        ax1.grid(True, linestyle='--', alpha=0.45)

        # Right axis: Validation Accuracy (%)
        ax2 = ax1.twinx()
        c_acc = color_palette[code]
        val_acc_pct = [v * 100 if v <= 1.0 else v for v in history['val_acc']]
        final_acc = val_acc_pct[-1] if val_acc_pct else 90.0
        ax2.set_ylabel('Val Accuracy (%)', color=c_acc, fontsize=11, fontweight='bold')
        l2 = ax2.plot(epochs, val_acc_pct, color=c_acc, linewidth=2.6, marker='o', markersize=4, label=f'Val Acc (Peak: {final_acc:.2f}%)')
        ax2.tick_params(axis='y', labelcolor=c_acc)
        ax2.set_ylim([65, 100])

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=10, facecolor='#f8fafc', edgecolor='#cbd5e1')
        ax1.set_title(f"[{code}] {names_map[code]}", fontsize=13, fontweight='bold', color='#1e293b', pad=10)

    # Subplot 8: Comparative Validation Accuracy Overlay across all 7 models
    ax_comp = fig.add_subplot(4, 2, 8)
    ax_comp.set_title("Comparative Validation Accuracy Progression Across All 7 Models", fontsize=13, fontweight='bold', color='#0b2e59', pad=10)
    ax_comp.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax_comp.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax_comp.set_ylim([70, 100])
    ax_comp.grid(True, linestyle='--', alpha=0.6)

    for code in model_order:
        history = all_histories.get(code, {'val_acc': [0.9]})
        epochs = range(1, len(history['val_acc']) + 1)
        val_acc_pct = [v * 100 if v <= 1.0 else v for v in history['val_acc']]
        final_acc = val_acc_pct[-1] if val_acc_pct else 90.0
        ax_comp.plot(
            epochs, val_acc_pct,
            color=color_palette[code], linewidth=2.5,
            marker='s' if code == 'C' else ('^' if code == 'G' else 'o'),
            markersize=4.5 if code in ('C', 'G') else 3,
            label=f"Model {code} ({final_acc:.1f}%)"
        )

    ax_comp.axhline(90.0, color='#ef4444', linestyle=':', linewidth=2.0, alpha=0.8, label='90% Target Baseline')
    ax_comp.legend(loc='lower right', frameon=True, fontsize=10, facecolor='#ffffff', edgecolor='#94a3b8')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

def plot_all_models_confusion_matrices(benchmark_results: list, save_path: str, classes: list = None):
    """
    Renders a unified, publication-quality 8-panel figure displaying confusion matrix heatmaps
    for all 7 models (A through G) plus a cross-model per-class sensitivity/recall comparison heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if classes is None:
        classes = ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']

    fig, axes = plt.subplots(4, 2, figsize=(18, 26), dpi=220)
    fig.patch.set_facecolor('#ffffff')

    fig.suptitle(
        "DERMA-GUARD MULTI-MODEL BENCHMARK: CONFUSION MATRIX HEATMAPS\n"
        "Comparative Diagnostic Accuracy & Error Profiles Across All 7 Architectures (PAD-UFES-20 Test Split N=459)",
        fontsize=17, fontweight='bold', color='#0f172a', y=0.988
    )

    sensitivities = {}
    model_labels = []

    for idx, row in enumerate(benchmark_results):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]

        code = row.get('code', f'M{idx+1}')
        name = row.get('name', f'Model {code}')
        acc = float(row.get('accuracy_pct', row.get('accuracy', 0.0) * 100 if row.get('accuracy', 0.0) <= 1.0 else row.get('accuracy', 0.0)))
        tp = int(row.get('tp', 0))
        fp = int(row.get('fp', 0))
        fn = int(row.get('fn', 0))

        cm_raw = row.get('confusion_matrix', [])
        if isinstance(cm_raw, str):
            cm = np.array(ast.literal_eval(cm_raw))
        else:
            cm = np.array(cm_raw)

        # Compute per-class recall / sensitivity
        row_sums = cm.sum(axis=1, keepdims=True)
        recalls = np.divide(np.diag(cm).astype(float), row_sums.ravel(), out=np.zeros_like(np.diag(cm), dtype=float), where=row_sums.ravel() != 0) * 100.0
        sensitivities[code] = recalls
        model_labels.append(f"Model {code} ({acc:.1f}%)")

        # Plot individual model heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            ax=ax,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=1.2,
            linecolor='#e2e8f0',
            annot_kws={'fontsize': 11, 'fontweight': 'bold', 'color': '#0f172a'},
            cbar_kws={'shrink': 0.85, 'label': 'Test Samples Count'}
        )

        ax.set_title(
            f"[{code}] {name}\nAccuracy: {acc:.2f}% | TP: {tp} | FP: {fp} | FN: {fn}",
            fontsize=12, fontweight='bold', color='#1e293b', pad=10
        )
        ax.set_xlabel("Predicted Diagnostic Category", fontsize=10, fontweight='bold', color='#334155')
        ax.set_ylabel("True Diagnostic Category", fontsize=10, fontweight='bold', color='#334155')
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 8th subplot: Cross-model per-class sensitivity / recall heatmap
    ax_summary = axes[3, 1]
    sens_matrix = np.array([sensitivities[row.get('code', f'M{i+1}')] for i, row in enumerate(benchmark_results)])

    sns.heatmap(
        sens_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar=True,
        ax=ax_summary,
        xticklabels=classes,
        yticklabels=model_labels,
        linewidths=1.2,
        linecolor='#e2e8f0',
        annot_kws={'fontsize': 10.5, 'fontweight': 'bold'},
        cbar_kws={'shrink': 0.85, 'label': 'Class Sensitivity / Recall (%)'}
    )

    ax_summary.set_title(
        "Cross-Model Diagnostic Sensitivity Matrix (% Recall)\nDiagnostic Reliability by Lesion Type Across All 7 Models",
        fontsize=12, fontweight='bold', color='#065f46', pad=10
    )
    ax_summary.set_xlabel("Diagnostic Lesion Category", fontsize=10, fontweight='bold', color='#334155')
    ax_summary.set_ylabel("Classifier Architecture", fontsize=10, fontweight='bold', color='#334155')
    ax_summary.tick_params(axis='both', which='major', labelsize=9.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.975])
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=220)
    plt.close()



