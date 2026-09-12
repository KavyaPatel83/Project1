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
    print(f"\n  +-- Performance Evaluation for Model {code} ({name}) --+")
    print(f"  | Accuracy          : {metrics['accuracy_pct']:6.2f}%")
    print(f"  | Precision (Macro) : {metrics['precision_macro']:6.4f}  | Precision (Weighted) : {metrics['precision_weighted']:6.4f}")
    print(f"  | Recall (Macro)    : {metrics['recall_macro']:6.4f}  | Recall (Weighted)    : {metrics['recall_weighted']:6.4f}")
    print(f"  | F1-Score (Macro)  : {metrics['f1_macro']:6.4f}  | F1-Score (Weighted)  : {metrics['f1_weighted']:6.4f}")
    print(f"  | ROC-AUC (Macro)   : {metrics['roc_auc_macro']:6.4f}  | ROC-AUC (Weighted)   : {metrics['roc_auc_weighted']:6.4f}")
    print(f"  | Conf Matrix (Agg) : TP={tp:<4} | TN={tn:<5} | FP={fp:<3} | FN={fn:<3}")
    print(f"  +-------------------------------------------------------------+", flush=True)

def print_comparison_table(results_list: list):
    """
    Prints a rich ASCII comparison table of all 7 models including TP, TN, FP, FN, and Dataset Size.
    """
    print("\n" + "=" * 126)
    print("                               FINAL MULTIMODAL BENCHMARK COMPARISON TABLE")
    print("=" * 126)
    header = f"{'Code':<5} | {'Model Name':<19} | {'Modality':<22} | {'Accuracy':<9} | {'F1-Macro':<8} | {'ROC-AUC':<8} | {'TP':<4} | {'FP':<3} | {'FN':<3} | {'Dataset Size (Train/Val/Test)':<26}"
    print(header)
    print("-" * 126)

    best_acc = -1.0
    best_code = None
    for res in results_list:
        code = res['code']
        name = res['name'][:19]
        mod = res['modality'][:22]
        acc = f"{res['accuracy_pct']:6.2f}%"
        f1 = f"{res['f1_macro']:6.4f}"
        auc = f"{res['roc_auc_macro']:6.4f}"
        tp = res.get('tp', '-')
        fp = res.get('fp', '-')
        fn = res.get('fn', '-')
        d_size = "N=2298 (1953/229/459)"
        print(f"{code:<5} | {name:<19} | {mod:<22} | {acc:<9} | {f1:<8} | {auc:<8} | {tp:<4} | {fp:<3} | {fn:<3} | {d_size:<26}")
        if res['accuracy_pct'] > best_acc:
            best_acc = res['accuracy_pct']
            best_code = code

    print("=" * 126)
    print(f"[BEST MODEL] BEST PERFORMING MODEL: Model {best_code} with Accuracy of {best_acc:.2f}%")
    print("=" * 126 + "\n", flush=True)

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
