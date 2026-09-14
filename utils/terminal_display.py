import sys

def print_header(title: str, subtitle: str = ""):
    border = "=" * 80
    print("\n" + border)
    print(f"   {title.upper()}")
    if subtitle:
        print(f"   {subtitle}")
    print(border + "\n", flush=True)

def print_section(title: str):
    print(f"\n---> {title}", flush=True)

def print_model_banner(code: str, name: str, modality: str):
    print("\n" + "-" * 75)
    print(f"  [MODEL {code}] : {name}")
    print(f"  Modality Configuration : {modality}")
    print("-" * 75, flush=True)

def print_metrics_summary(code: str, name: str, metrics: dict):
    tp = metrics.get('tp', 'N/A')
    tn = metrics.get('tn', 'N/A')
    fp = metrics.get('fp', 'N/A')
    fn = metrics.get('fn', 'N/A')
    sens_m = metrics.get('sensitivity_macro', metrics.get('recall_macro', 0.0))
    sens_w = metrics.get('sensitivity_weighted', metrics.get('recall_weighted', 0.0))
    spec_m = metrics.get('specificity_macro', 0.0)
    spec_w = metrics.get('specificity_weighted', 0.0)
    auc_m = metrics.get('auc_macro', metrics.get('roc_auc_macro', 0.0))
    auc_w = metrics.get('auc_weighted', metrics.get('roc_auc_weighted', 0.0))
    auprc_m = metrics.get('auprc_macro', 0.0)
    auprc_w = metrics.get('auprc_weighted', 0.0)

    print(f"\n  +-- Performance Evaluation for Model {code} ({name}) --+")
    print(f"  | Accuracy            : {metrics['accuracy_pct']:6.2f}%")
    print(f"  | Sensitivity (Recall): Macro = {sens_m:6.4f}  | Weighted = {sens_w:6.4f}")
    print(f"  | Specificity (TNR)   : Macro = {spec_m:6.4f}  | Weighted = {spec_w:6.4f}")
    print(f"  | AUC (ROC-AUC)       : Macro = {auc_m:6.4f}  | Weighted = {auc_w:6.4f}")
    print(f"  | AUPRC (PR-AUC)      : Macro = {auprc_m:6.4f}  | Weighted = {auprc_w:6.4f}")
    print(f"  | Precision           : Macro = {metrics['precision_macro']:6.4f}  | Weighted = {metrics['precision_weighted']:6.4f}")
    print(f"  | F1-Score            : Macro = {metrics['f1_macro']:6.4f}  | Weighted = {metrics['f1_weighted']:6.4f}")
    print(f"  | Conf Matrix (Agg)   : TP = {tp:<4} | TN = {tn:<5} | FP = {fp:<3} | FN = {fn:<3}")
    print(f"  +-----------------------------------------------------------------------------+", flush=True)

def print_comparison_table(results_list: list):
    """
    Prints a rich ASCII comparison table of all 7 models including Precision, Sensitivity, Specificity, F1-Score, AUC, AUPRC, TP, FP, FN, and Dataset Size.
    """
    width = 175
    print("\n" + "=" * width)
    print("                                                        FINAL MULTIMODAL BENCHMARK COMPARISON TABLE")
    print("=" * width)
    header = f"{'Code':<5} | {'Model Name':<18} | {'Modality':<20} | {'Accuracy':<9} | {'Precision':<10} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Score':<9} | {'AUC':<8} | {'AUPRC':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Dataset Split':<24}"
    print(header)
    print("-" * width)

    best_acc = -1.0
    best_code = None
    for res in results_list:
        code = res['code']
        name = res['name'][:18]
        mod = res['modality'][:20]
        acc = f"{res['accuracy_pct']:6.2f}%"
        prec = f"{res.get('precision_macro', 0.0):6.4f}"
        sens = f"{res.get('sensitivity_macro', res.get('recall_macro', 0.0)):6.4f}"
        spec = f"{res.get('specificity_macro', 0.0):6.4f}"
        f1 = f"{res.get('f1_macro', 0.0):6.4f}"
        auc = f"{res.get('auc_macro', res.get('roc_auc_macro', 0.0)):6.4f}"
        auprc = f"{res.get('auprc_macro', 0.0):6.4f}"
        tp = res.get('tp', '-')
        fp = res.get('fp', '-')
        fn = res.get('fn', '-')
        d_size = "N=2298 (1953/229/459)"
        print(f"{code:<5} | {name:<18} | {mod:<20} | {acc:<9} | {prec:<10} | {sens:<11} | {spec:<11} | {f1:<9} | {auc:<8} | {auprc:<8} | {tp:<4} | {fp:<4} | {fn:<4} | {d_size:<24}")
        if res['accuracy_pct'] > best_acc:
            best_acc = res['accuracy_pct']
            best_code = code

    print("=" * width)
    print(f"[BEST MODEL] BEST PERFORMING MODEL: Model {best_code} with Accuracy of {best_acc:.2f}%")
    print("=" * width + "\n", flush=True)

def print_evidence_table(evidence_records: list):
    """
    Prints sample evidence manager breakdown table.
    """
    print("\n" + "=" * 88)
    print("           SAMPLE DERMA-GUARD EVIDENCE PROFILE & DYNAMIC FUSION WEIGHTS")
    print("=" * 88)
    header = f"{'Modality':<12} | {'Quality (Q)':<12} | {'Reliability (R)':<16} | {'Utility (U)':<12} | {'Consist (C)':<12} | {'Evidence (E)':<12} | {'Weight (w)':<10}"
    print(header)
    print("-" * 88)
    for rec in evidence_records:
        m = rec.get('modality', 'N/A')
        q = f"{rec.get('q', 0.0):.3f}"
        r = f"{rec.get('r', 0.0):.3f}"
        u = f"{rec.get('u', 0.0):.3f}"
        c = f"{rec.get('c', 0.0):.3f}"
        e = f"{rec.get('e', 0.0):.3f}"
        w = f"{rec.get('w', 0.0)*100:.1f}%"
        print(f"{m:<12} | {q:<12} | {r:<16} | {u:<12} | {c:<12} | {e:<12} | {w:<10}")
    print("=" * 88 + "\n", flush=True)
