"""
Generate Clean, Publication-Grade Architecture Diagram matching the reference diagram exactly.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_clean_architecture_diagram(output_png, output_jpg):
    # Canvas size: 26 x 13.5 inches at 300 DPI = 7800 x 4050 px
    fig = plt.figure(figsize=(26, 13.5), dpi=300, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    # Color definitions - exact match to clean clinical diagram style
    BG_CANVAS = '#ffffff'
    
    # Card outer themes
    COLOR_INPUTS_BG = '#f8fafc'
    COLOR_INPUTS_BORDER = '#3b82f6'
    COLOR_INPUTS_TEXT = '#1e3a8a'
    
    COLOR_ENCODERS_BG = '#f0fdf4'
    COLOR_ENCODERS_BORDER = '#22c55e'
    COLOR_ENCODERS_TEXT = '#14532d'

    COLOR_EVIDENCE_BG = '#fffbeb'
    COLOR_EVIDENCE_BORDER = '#f59e0b'
    COLOR_EVIDENCE_TEXT = '#78350f'

    COLOR_FUSION_BG = '#f0f9ff'
    COLOR_FUSION_BORDER = '#0284c7'
    COLOR_FUSION_TEXT = '#0c4a6e'

    COLOR_PRED_BG = '#faf5ff'
    COLOR_PRED_BORDER = '#a855f7'
    COLOR_PRED_TEXT = '#581c87'

    COLOR_DECISION_BG = '#fef2f2'
    COLOR_DECISION_BORDER = '#ef4444'
    COLOR_DECISION_TEXT = '#7f1d1d'

    COLOR_SAFETY_BG = '#f0f9ff'
    COLOR_SAFETY_BORDER = '#38bdf8'
    COLOR_SAFETY_TEXT = '#0369a1'

    COLOR_OUTPUT_BG = '#fff7ed'
    COLOR_OUTPUT_BORDER = '#f97316'
    COLOR_OUTPUT_TEXT = '#7c2d12'

    COLOR_RETRIEVAL_BG = '#f8fafc'
    COLOR_RETRIEVAL_BORDER = '#64748b'
    COLOR_RETRIEVAL_TEXT = '#1e293b'

    # Box drawer
    def draw_container(x, y, w, h, title="", subtitle="", bg='#ffffff', border='#94a3b8', title_color='#0f172a', radius=0.25, lw=1.5):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=border,
            facecolor=bg,
            zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.35, title, fontsize=11, fontweight='bold', color=title_color, ha='center', va='center', zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.65, subtitle, fontsize=8.5, color='#64748b', ha='center', va='center', zorder=3)
        return box

    def draw_subcard(x, y, w, h, title="", lines=None, bg='#ffffff', border='#cbd5e1', title_color='#1e293b', radius=0.18, lw=1.2, align='left'):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=border,
            facecolor=bg,
            zorder=3
        )
        ax.add_patch(box)
        curr_y = y + h - 0.3
        if title:
            if align == 'center':
                ax.text(x + w/2, curr_y, title, fontsize=9.5, fontweight='bold', color=title_color, ha='center', va='center', zorder=4)
            else:
                ax.text(x + 0.2, curr_y, title, fontsize=9.5, fontweight='bold', color=title_color, ha='left', va='center', zorder=4)
            curr_y -= 0.32
        
        if lines:
            line_spacing = 0.24
            for line in lines:
                if align == 'center':
                    ax.text(x + w/2, curr_y, line, fontsize=8.0, color='#334155', ha='center', va='center', zorder=4)
                else:
                    ax.text(x + 0.2, curr_y, line, fontsize=8.0, color='#334155', ha='left', va='center', zorder=4)
                curr_y -= line_spacing
        return box

    # Arrow drawer
    def draw_arrow(start, end, text="", color='#334155', lw=1.5, rad=0.0, ls='-', style='-|>', text_offset=(0, 0.15)):
        arrow = FancyArrowPatch(
            posA=start, posB=end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=color,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=14,
            zorder=5
        )
        ax.add_patch(arrow)
        if text:
            mid_x = (start[0] + end[0]) / 2 + text_offset[0]
            mid_y = (start[1] + end[1]) / 2 + text_offset[1]
            ax.text(mid_x, mid_y, text, fontsize=8.5, fontweight='bold', color=color,
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', edgecolor='none', alpha=0.85))

    # ==========================================
    # 1. MULTIMODAL INPUTS (Top Left)
    # ==========================================
    draw_container(0.6, 6.2, 3.6, 6.8, title="MULTIMODAL INPUTS", bg='#f8fafc', border=COLOR_INPUTS_BORDER, title_color=COLOR_INPUTS_TEXT, lw=1.8)
    
    draw_subcard(0.8, 10.7, 3.2, 1.8, title="Visual Inputs",
                 lines=["• Clinical Image", "  (Smartphone)", "• Dermoscopy Image"],
                 border='#93c5fd', title_color='#1e40af')

    draw_subcard(0.8, 8.5, 3.2, 1.9, title="Structured Metadata",
                 lines=["• Age, Sex, Skin Type", "• Location, History,", "  and other attributes"],
                 border='#93c5fd', title_color='#1e40af')

    draw_subcard(0.8, 6.5, 3.2, 1.7, title="Patient Language",
                 lines=["• Free-text description", "• (Synthetic or real)"],
                 border='#93c5fd', title_color='#1e40af')

    # ==========================================
    # 2. MODALITY ENCODERS
    # ==========================================
    draw_container(4.8, 6.2, 3.6, 6.8, title="MODALITY ENCODERS", bg='#f0fdf4', border=COLOR_ENCODERS_BORDER, title_color=COLOR_ENCODERS_TEXT, lw=1.8)

    draw_subcard(5.0, 11.2, 3.2, 1.3, title="Image Encoder",
                 lines=["(Swin Transformer)"], align='center',
                 border='#86efac', title_color='#166534')

    draw_subcard(5.0, 9.6, 3.2, 1.3, title="Dermoscopy Encoder",
                 lines=["(Swin Transformer)"], align='center',
                 border='#86efac', title_color='#166534')

    draw_subcard(5.0, 8.0, 3.2, 1.3, title="Metadata Encoder",
                 lines=["(MLP / MetaBlock)"], align='center',
                 border='#86efac', title_color='#166534')

    draw_subcard(5.0, 6.4, 3.2, 1.3, title="Language Encoder",
                 lines=["(BioBERT / Clinical BERT)"], align='center',
                 border='#86efac', title_color='#166534')

    # Arrows from Inputs to Encoders
    draw_arrow((4.0, 11.7), (5.0, 11.8), color='#334155', lw=1.5)
    draw_arrow((4.0, 11.4), (5.0, 10.3), color='#334155', lw=1.5)
    draw_arrow((4.0, 9.4), (5.0, 8.7), color='#334155', lw=1.5)
    draw_arrow((4.0, 7.3), (5.0, 7.1), color='#334155', lw=1.5)

    # ==========================================
    # 3. EVIDENCE MANAGER (Per Modality & Per Case)
    # ==========================================
    draw_container(9.0, 6.2, 6.8, 6.8, title="EVIDENCE MANAGER (Per Modality & Per Case)", bg='#fffbeb', border=COLOR_EVIDENCE_BORDER, title_color=COLOR_EVIDENCE_TEXT, lw=1.8)

    # Top-Left: Quality Estimator
    draw_subcard(9.2, 9.5, 3.1, 2.7, title="Quality Estimator",
                 lines=["• Image quality (blur, exposure)", "• View adequacy", "• Completeness"],
                 align='center', border='#fcd34d', title_color='#92400e')

    # Top-Right: Reliability Estimator
    draw_subcard(12.5, 9.5, 3.1, 2.7, title="Reliability Estimator",
                 lines=["• Modality reliability", "• Source reliability", "• Noise / Artifact score"],
                 align='center', border='#fcd34d', title_color='#92400e')

    # Bottom-Left: Counterfactual Utility Estimator
    draw_subcard(9.2, 7.7, 3.1, 1.6, title="Counterfactual Utility Estimator",
                 lines=["• Patient-specific utility", "• How much this modality", "  can change the decision"],
                 align='center', border='#fcd34d', title_color='#92400e')

    # Bottom-Right: Cross-Modal Consistency Estimator
    draw_subcard(12.5, 7.7, 3.1, 1.6, title="Cross-Modal Consistency Estimator",
                 lines=["• Agreement with other", "  modalities", "• Conflict score"],
                 align='center', border='#fcd34d', title_color='#92400e')

    # Evidence Summary (Per Case)
    draw_subcard(9.2, 6.4, 6.4, 1.1, title="Evidence Summary (Per Case)",
                 lines=["• Aggregated scores for all modalities  • Identify weak / missing / conflicting evidence"],
                 align='center', border='#fcd34d', title_color='#92400e')

    # Arrow from Encoders to Evidence Manager
    draw_arrow((8.4, 10.5), (9.0, 10.5), color='#334155', lw=1.6)
    draw_arrow((8.4, 7.8), (9.0, 8.2), color='#334155', lw=1.6)

    # ==========================================
    # 4. ADAPTIVE FUSION MODULE (Evidence-Aware)
    # ==========================================
    draw_container(16.4, 8.5, 3.8, 4.5, title="ADAPTIVE FUSION MODULE", subtitle="(Evidence-Aware)",
                   bg='#f0f9ff', border=COLOR_FUSION_BORDER, title_color=COLOR_FUSION_TEXT, lw=1.8)

    draw_subcard(16.7, 10.0, 3.2, 1.5, title="Gated Fusion",
                 lines=["(Dynamic Weights)"], align='center',
                 border='#7dd3fc', title_color='#0369a1')

    # Fused Representation label below Gated Fusion
    draw_arrow((18.3, 10.0), (18.3, 9.4), color='#0284c7', lw=1.6)
    ax.text(18.3, 9.0, "Fused Representation", fontsize=8.5, fontweight='bold', color='#0f172a', ha='center', va='center', zorder=4)

    # Arrow from Evidence Manager to Adaptive Fusion
    draw_arrow((15.8, 10.5), (16.4, 10.5), color='#334155', lw=1.6)

    # ==========================================
    # 5. PREDICTION & UNCERTAINTY ESTIMATION
    # ==========================================
    draw_container(20.8, 8.5, 4.6, 4.5, title="PREDICTION & UNCERTAINTY", subtitle="ESTIMATION",
                   bg='#faf5ff', border=COLOR_PRED_BORDER, title_color=COLOR_PRED_TEXT, lw=1.8)

    draw_subcard(21.1, 10.8, 4.0, 1.4, title="Classifier Head",
                 lines=["(Skin Lesion Classes)"], align='center',
                 border='#d8b4fe', title_color='#6b21a8')

    draw_subcard(21.1, 8.9, 4.0, 1.4, title="Uncertainty Estimation",
                 lines=["(Conformal Prediction)"], align='center',
                 border='#d8b4fe', title_color='#6b21a8')

    # Arrow from Adaptive Fusion to Prediction
    draw_arrow((20.2, 10.2), (20.8, 10.2), color='#334155', lw=1.6)

    # ==========================================
    # 6. DECISION MODULE (Evidence Sufficiency Check)
    # ==========================================
    draw_container(18.0, 3.8, 7.4, 4.0, title="DECISION MODULE", subtitle="(Evidence Sufficiency Check)",
                   bg='#fef2f2', border=COLOR_DECISION_BORDER, title_color=COLOR_DECISION_TEXT, lw=1.8)

    # Decision Question Box
    draw_subcard(19.8, 5.8, 3.8, 1.0, title="Is Evidence Sufficient?", align='center',
                 bg='#ffffff', border='#fca5a5', title_color='#991b1b')

    # Branch YES -> PREDICT
    draw_subcard(18.3, 4.2, 3.2, 1.2, title="PREDICT",
                 lines=["(Provide Diagnosis,", "Confidence & Explanation)"], align='center',
                 bg='#f0fdf4', border='#86efac', title_color='#166534')

    # Branch NO -> ACQUIRE / ABSTAIN
    draw_subcard(21.9, 4.2, 3.2, 1.2, title="ACQUIRE / ABSTAIN",
                 lines=["(Request More Evidence", "or Abstain from Prediction)"], align='center',
                 bg='#fff1f2', border='#fecdd3', title_color='#9f1239')

    # Arrows inside Decision Module
    draw_arrow((23.1, 8.5), (23.1, 7.8), color='#334155', lw=1.6)
    draw_arrow((21.7, 5.8), (19.9, 5.4), color='#16a34a', lw=1.5, text="Yes", text_offset=(-0.25, 0.12))
    draw_arrow((21.7, 5.8), (23.5, 5.4), color='#dc2626', lw=1.5, text="No", text_offset=(0.25, 0.12))

    # ==========================================
    # 7. SIMULATED EVIDENCE ACQUISITION POLICY
    # ==========================================
    draw_container(10.2, 3.8, 4.8, 2.0, title="SIMULATED EVIDENCE", subtitle="ACQUISITION POLICY",
                   bg='#f0f9ff', border='#38bdf8', title_color='#0369a1', lw=1.5)
    ax.text(12.6, 4.8, "• Which modality to acquire?\n• Next-best-evidence strategy\n• Cost-aware acquisition",
            fontsize=8.0, color='#334155', ha='center', va='top', zorder=4)

    # Arrow from ACQUIRE/ABSTAIN to Acquisition Policy (Dashed)
    draw_arrow((21.9, 4.6), (15.0, 4.6), color='#64748b', lw=1.4, ls='--')

    # Arrow from Acquisition Policy up to Evidence Manager
    draw_arrow((12.6, 5.8), (12.6, 6.2), color='#334155', lw=1.5)

    # ==========================================
    # 8. EXPLAINABILITY MODULE
    # ==========================================
    draw_container(3.2, 3.5, 4.8, 2.1, title="EXPLAINABILITY MODULE",
                   bg='#faf5ff', border='#c084fc', title_color='#6b21a8', lw=1.5)
    ax.text(5.6, 4.8, "• Gradient-based attention maps\n• Highlight important regions and modalities\n• Modal contribution analysis",
            fontsize=8.0, color='#334155', ha='center', va='top', zorder=4)

    # Arrow from Modality Encoders down to Explainability
    draw_arrow((5.6, 6.2), (5.6, 5.6), color='#334155', lw=1.5)

    # ==========================================
    # 9. SIMILAR CASE RETRIEVAL (GROUNDING)
    # ==========================================
    draw_container(1.4, 0.6, 4.8, 2.3, title="SIMILAR CASE RETRIEVAL", subtitle="(GROUNDING)",
                   bg='#f8fafc', border=COLOR_RETRIEVAL_BORDER, title_color=COLOR_RETRIEVAL_TEXT, lw=1.5)
    ax.text(3.8, 1.8, "• Retrieve similar past cases\n• Provide reference outcomes\n• Support clinician trust",
            fontsize=8.0, color='#334155', ha='center', va='top', zorder=4)

    # ==========================================
    # 10. FINAL OUTPUT TO USER / CLINICIAN
    # ==========================================
    draw_container(8.0, 0.5, 6.8, 2.5, title="FINAL OUTPUT TO USER / CLINICIAN",
                   bg='#fff7ed', border=COLOR_OUTPUT_BORDER, title_color=COLOR_OUTPUT_TEXT, lw=1.8)
    ax.text(8.3, 2.3, "• Predicted diagnosis (if sufficient)\n• Confidence / Prediction set\n• Explanation (visual + textual)\n• Evidence sufficiency status\n• Recommended next action (if any)",
            fontsize=8.2, color='#334155', ha='left', va='top', zorder=4)

    # ==========================================
    # 11. RULE + LLM VERIFICATION & SAFETY LAYER
    # ==========================================
    draw_container(16.6, 0.5, 5.8, 2.5, title="RULE + LLM VERIFICATION & SAFETY LAYER",
                   bg='#f0f9ff', border=COLOR_SAFETY_BORDER, title_color=COLOR_SAFETY_TEXT, lw=1.5)
    ax.text(16.9, 2.3, "• Rule-based leakage detection\n• Hallucination check\n• Consistency with metadata\n• Safe clinical statement generation",
            fontsize=8.2, color='#334155', ha='left', va='top', zorder=4)

    # Dashed arrow from PREDICT down to Safety Layer
    draw_arrow((19.5, 4.2), (19.5, 3.0), color='#64748b', lw=1.3, ls='--')
    # Dashed arrow from ACQUIRE down to Safety Layer
    draw_arrow((23.0, 4.2), (21.5, 3.0), color='#64748b', lw=1.3, ls='--')

    # Arrow from Safety Layer to Final Output
    draw_arrow((16.6, 1.7), (14.8, 1.7), color='#334155', lw=1.6)

    # Arrow from Similar Case Retrieval to Final Output
    draw_arrow((6.2, 1.7), (8.0, 1.7), color='#334155', lw=1.6)

    # Save outputs
    plt.savefig(output_png, format='png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(output_jpg, format='jpg', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved clean architecture diagram to:\n  - {output_png}\n  - {output_jpg}")

if __name__ == "__main__":
    base_dir = r"d:\VIT BOOKS\PROJECT 1\Project"
    saved_models_dir = os.path.join(base_dir, "saved_models")
    
    png_path = os.path.join(saved_models_dir, "derma_guard_architecture_clean.png")
    jpg_path = os.path.join(saved_models_dir, "derma_guard_architecture_clean.jpg")
    
    create_clean_architecture_diagram(png_path, jpg_path)

    # Also update the primary workflow images so all references show this clean style!
    primary_png = os.path.join(saved_models_dir, "derma_guard_architecture_workflow.png")
    primary_jpg = os.path.join(saved_models_dir, "derma_guard_architecture_workflow.jpg")
    create_clean_architecture_diagram(primary_png, primary_jpg)
