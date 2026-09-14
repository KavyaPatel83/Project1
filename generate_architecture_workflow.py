"""
Generate Publication-Quality DERMA-GUARD v4.0 Multi-Agent Architecture & Workflow Diagram
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_architecture_diagram(output_png_path, output_jpg_path):
    # Set canvas size: 28 x 17.5 inches at 300 DPI = 8400 x 5250 px
    fig = plt.figure(figsize=(28, 17.5), dpi=300, facecolor='#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 17.5)
    ax.axis('off')

    # Color Palette - Modern Clinical AI Dark Glassmorphism Theme
    BG_CARD = '#161b22'
    BORDER_COLOR = '#30363d'
    TITLE_COLOR = '#f0f6fc'
    TEXT_MUTED = '#8b949e'
    ACCENT_BLUE = '#388bfd'
    ACCENT_CYAN = '#39c5cf'
    ACCENT_PURPLE = '#a371f7'
    ACCENT_AMBER = '#d29922'
    ACCENT_GREEN = '#3fb950'
    ACCENT_RED = '#f85149'
    ACCENT_ORANGE = '#f0883e'

    # Helper: Draw rounded box
    def draw_card(x, y, w, h, title="", subtitle="", bg=BG_CARD, border=BORDER_COLOR, title_color=TITLE_COLOR, border_width=1.5, radius=0.25):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=border_width,
            edgecolor=border,
            facecolor=bg,
            zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(x + 0.35, y + h - 0.45, title, fontsize=12, fontweight='bold', color=title_color, zorder=3)
        if subtitle:
            ax.text(x + 0.35, y + h - 0.8, subtitle, fontsize=9, fontstyle='italic', color=TEXT_MUTED, zorder=3)
        return box

    # Helper: Draw arrow
    def draw_arrow(start, end, text="", color='#58a6ff', rad=0.0, lw=2.0, style='-|>', ls='-'):
        arrow = FancyArrowPatch(
            posA=start, posB=end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=color,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=16,
            zorder=4
        )
        ax.add_patch(arrow)
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + (0.15 if rad >= 0 else -0.15)
            ax.text(mid_x, mid_y, text, fontsize=8.5, fontweight='bold', color=color,
                    ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor=color, alpha=0.9, lw=0.8))

    # ==========================================
    # HEADER BANNER
    # ==========================================
    header_box = FancyBboxPatch((0.6, 16.0), 26.8, 1.1, boxstyle="round,pad=0,rounding_size=0.2",
                                facecolor='#161b22', edgecolor='#388bfd', linewidth=2.0, zorder=2)
    ax.add_patch(header_box)
    ax.text(14.0, 16.7, "DERMA-GUARD v4.0: MULTI-AGENT EVIDENCE-AWARE ARCHITECTURE & WORKFLOW",
            fontsize=19, fontweight='bold', color='#ffffff', ha='center', va='center', zorder=3)
    ax.text(14.0, 16.25, r"Parallel Specialist Agents • 5D Evidence Profiling ($Q, R, U, C, M, \mathrm{OOD}$) • Iterative EM Diagnosis-Adaptive Fusion • 4-Level Consensus • Hallucination Verification",
            fontsize=10.5, fontstyle='italic', color='#58a6ff', ha='center', va='center', zorder=3)

    # ==========================================
    # STAGE 1: MULTIMODAL INGESTION & PARALLEL SPECIALIST AGENTS
    # ==========================================
    # Outer Column Box
    draw_card(0.6, 0.6, 5.0, 15.1, "STAGE 1: INPUTS & SPECIALIST AGENTS", "Multimodal Ingestion & Feature Encoders",
              bg='#11151c', border='#1f6feb', border_width=2.0)

    # 1.1 Ingestion & Sanitization
    draw_card(0.9, 12.8, 4.4, 2.0, "1. Patient Ingestion & Sanitize", "MIME, Security, Aspect Ratio & Bounds", bg='#1c2128', border='#388bfd')
    ax.text(1.1, 13.7, "• Clinical Cutaneous Photography\n• Polarized Dermoscopy Images\n• Free-Text Clinical Narratives\n• Tabular Patient Metadata",
            fontsize=8.5, color='#c9d1d9', va='top', zorder=3)
    ax.text(1.1, 13.0, r"Sanitization: $32 \leq \mathrm{dim} \leq 4096$, Size < 10MB" + "\n" + r"Missingness Mask: $\mathbf{M} \in \{0, 1\}^4$",
            fontsize=8.0, color='#79c0ff', va='top', zorder=3)

    # 1.2 Vision Specialist Team
    draw_card(0.9, 9.1, 4.4, 3.4, "2. Vision Specialist Team", "Multi-Head Self-Attention ViT + CNN", bg='#161b22', border='#a371f7')
    ax.text(1.1, 11.5, "Constituent Visual Encoders:", fontsize=8.5, fontweight='bold', color='#d2a8ff', zorder=3)
    ax.text(1.1, 11.2, "• Swin-Tiny (Clinical & Dermoscopy)\n• ResNet152 Deep Feature Extractor\n• PanDerm Foundation Visual Backbone\n• DermLIP Zero-Shot Alignment",
            fontsize=8.0, color='#e6edf3', va='top', zorder=3)
    ax.text(1.1, 9.6, r"Representation:" + "\n" + r"$\mathbf{z}_{\mathrm{img}} \in \mathbb{R}^{2048}$ (ViT + Conv Manifold)",
            fontsize=8.5, fontweight='bold', color='#7ee787', va='top', zorder=3)

    # 1.3 Scribe Agent
    draw_card(0.9, 5.2, 4.4, 3.6, "3. Scribe Agent (Clinical NLP)", "Bio_ClinicalBERT & PTIS Evaluator", bg='#161b22', border='#39c5cf')
    ax.text(1.1, 7.8, "Clinical Text Processing Pipeline:", fontsize=8.5, fontweight='bold', color='#76e3ea', zorder=3)
    ax.text(1.1, 7.5, "• Patient History & Evolution Narrative\n• Bio_ClinicalBERT Tokenization (768-d)\n• Extraction: Itch, Bleed, Pain, Growth\n• PTIS Quality Scoring (0 to 1 scale)",
            fontsize=8.0, color='#e6edf3', va='top', zorder=3)
    ax.text(1.1, 5.7, r"Representation:" + "\n" + r"$\mathbf{z}_{\mathrm{text}} \in \mathbb{R}^{768}$, Quality $Q_{\mathrm{text}} = \mathrm{PTIS}$",
            fontsize=8.5, fontweight='bold', color='#7ee787', va='top', zorder=3)

    # 1.4 Metadata Specialist
    draw_card(0.9, 0.9, 4.4, 4.0, "4. Metadata Specialist Agent", "Residual MetaBlock MLP", bg='#161b22', border='#d29922')
    ax.text(1.1, 3.9, "Tabular Clinical Variables:", fontsize=8.5, fontweight='bold', color='#e3b341', zorder=3)
    ax.text(1.1, 3.6, "• Age, Gender, Lesion Site / Anatomy\n• Diameters (1 & 2), Fitzpatrick Type\n• Symptoms: Itch, Grew, Hurt, Elevation\n• Residual Skip-Connection Linear Layers",
            fontsize=8.0, color='#e6edf3', va='top', zorder=3)
    ax.text(1.1, 1.5, r"Representation:" + "\n" + r"$\mathbf{z}_{\mathrm{meta}} \in \mathbb{R}^{116}$, $Q_{\mathrm{meta}} = 1 - \mathrm{frac}_{\mathrm{missing}}$",
            fontsize=8.5, fontweight='bold', color='#7ee787', va='top', zorder=3)

    # Arrows inside Stage 1
    draw_arrow((3.1, 12.8), (3.1, 12.5), lw=1.5, color='#388bfd')
    draw_arrow((3.1, 12.5), (3.1, 9.1), rad=-0.25, lw=1.5, color='#a371f7')
    draw_arrow((3.1, 12.5), (3.1, 5.2), rad=0.25, lw=1.5, color='#39c5cf')
    draw_arrow((3.1, 12.5), (3.1, 0.9), rad=-0.35, lw=1.5, color='#d29922')

    # ==========================================
    # STAGE 2: DEVIL'S ADVOCATE & 5D EVIDENCE PROFILER
    # ==========================================
    draw_card(5.9, 0.6, 5.1, 15.1, "STAGE 2: DEVIL'S ADVOCATE PROFILER", "5D Evidence Metrics & Stage 1 Safety Gate",
              bg='#11151c', border='#39c5cf', border_width=2.0)

    # 2.1 5D Evidence Evaluation Metrics
    draw_card(6.2, 7.8, 4.5, 7.0, "Evidence Metric Computation", "Independent Evaluation of Every Modality", bg='#161b22', border='#58a6ff')
    ax.text(6.4, 14.0, r"Quality Score ($Q_m \in [0, 1]$):", fontsize=8.5, fontweight='bold', color='#79c0ff', zorder=3)
    ax.text(6.4, 13.4, r"$Q_{\mathrm{img}} = \mathrm{MLP}([\mathrm{blur}, \mathrm{contrast}, \mathrm{resol}, \mathrm{occl}])$" + "\n" + r"$Q_{\mathrm{text}} = \mathrm{PTIS}(x), \quad Q_{\mathrm{meta}} = 1 - \mathrm{missing\_ratio}$",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    ax.text(6.4, 12.3, r"Reliability Score ($R_m \in [0, 1]$):", fontsize=8.5, fontweight='bold', color='#7ee787', zorder=3)
    ax.text(6.4, 11.7, r"$R_m = \sigma(\mathrm{MLP}([Q_m, H_m, S_m, A_m]))$" + "\n" + r"$H_m$: Predictive Entropy, $S_m$: TTA Invariance" + "\n" + r"$A_m$: Mean Cosine Agreement with Peer Models",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    ax.text(6.4, 10.4, r"Cross-Modal Consistency ($C \in [0, 1]$):", fontsize=8.5, fontweight='bold', color='#d2a8ff', zorder=3)
    ax.text(6.4, 9.8, r"$C(x) = 1 - \frac{2}{|M|(|M|-1)} \sum_{i < j} D_{\mathrm{JS}}(p_i, p_j)$" + "\n(Pairwise Jensen-Shannon Divergence)",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    ax.text(6.4, 8.8, "Out-of-Distribution (OOD) Detector:", fontsize=8.5, fontweight='bold', color='#ff7b72', zorder=3)
    ax.text(6.4, 8.2, r"$d_{\mathrm{Maha}}(z) = \min_k (z - \mu_k)^T \Sigma_k^{-1} (z - \mu_k)$" + "\n(Class-conditional Mahalanobis metric)",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    # 2.2 Stage 1 Verification & Safety Gate
    draw_card(6.2, 0.9, 4.5, 6.6, "Stage 1 Verification Gate", "Pre-Fusion Sanity & Safety Decisions", bg='#1c2128', border='#ff7b72')
    
    # Branch boxes inside Stage 1 Gate
    # Box A: OOD Escalate
    draw_card(6.4, 5.4, 4.1, 1.4, "OOD Anomaly Gate", "", bg='#21262d', border='#f85149', radius=0.15)
    ax.text(6.6, 6.2, r"IF $d_{\mathrm{Maha}} > \tau_{\mathrm{OOD}}$:", fontsize=8.0, fontweight='bold', color='#ff7b72', zorder=3)
    ax.text(6.6, 5.7, "→ ESCALATE to Dermatologist Board", fontsize=8.0, color='#ffa198', zorder=3)

    # Box B: Missing Modality
    draw_card(6.4, 3.8, 4.1, 1.4, "Evidence Sufficiency Check", "", bg='#21262d', border='#d29922', radius=0.15)
    ax.text(6.6, 4.6, r"IF $\sum \mathbf{M} == 0$ (No Active Modalities):", fontsize=8.0, fontweight='bold', color='#e3b341', zorder=3)
    ax.text(6.6, 4.1, "→ ABSTAIN ('Insufficient Evidence')", fontsize=8.0, color='#f2cc60', zorder=3)

    # Box C: Quality Pruning & Conflict
    draw_card(6.4, 1.2, 4.1, 2.3, "Pruning & Conflict Acquisition", "", bg='#21262d', border='#a371f7', radius=0.15)
    ax.text(6.6, 2.9, r"IF $Q_m < 0.2$:", fontsize=8.0, fontweight='bold', color='#d2a8ff', zorder=3)
    ax.text(6.6, 2.6, r"→ Prune Modality: Set $w_m = 0$", fontsize=8.0, color='#e6edf3', zorder=3)
    ax.text(6.6, 2.1, r"IF $C < 0.3$ AND No Consensus:", fontsize=8.0, fontweight='bold', color='#f0883e', zorder=3)
    ax.text(6.6, 1.6, "→ ACQUIRE ('Trigger Targeted Acquisition')", fontsize=8.0, color='#ffc680', zorder=3)

    # Arrows from Stage 1 to Stage 2
    draw_arrow((5.3, 10.8), (6.2, 11.3), text="z_img", color='#a371f7', lw=1.8)
    draw_arrow((5.3, 7.0), (6.2, 10.0), text="z_text", color='#39c5cf', lw=1.8)
    draw_arrow((5.3, 2.9), (6.2, 8.5), text="z_meta", color='#d29922', lw=1.8)
    draw_arrow((8.4, 7.8), (8.4, 7.5), text="Profile Pass", color='#3fb950', lw=2.0)

    # ==========================================
    # STAGE 3: ITERATIVE DIAGNOSIS-ADAPTIVE FUSION (EM-STYLE)
    # ==========================================
    draw_card(11.3, 0.6, 5.4, 15.1, "STAGE 3: ITERATIVE DYNAMIC FUSION", "Diagnosis-Adaptive EM Weight Refinement",
              bg='#11151c', border='#d29922', border_width=2.0)

    # 3.1 Initial Evidence Weighting
    draw_card(11.6, 12.3, 4.8, 2.5, "Step 0: Initial Weight Assignment", "Unconditioned Evidence Synthesis", bg='#161b22', border='#d29922')
    ax.text(11.8, 13.9, "Evidence Prior Calculation:", fontsize=8.5, fontweight='bold', color='#f2cc60', zorder=3)
    ax.text(11.8, 13.5, r"$\mathbf{w}^{(0)} = \mathrm{softmax}(\alpha \cdot \mathbf{Q} \odot \mathbf{R})$", fontsize=10.0, color='#ffffff', zorder=3)
    ax.text(11.8, 12.8, r"• Modality Quality $\odot$ Reliability" + "\n" + r"• Mask missing modalities: $\tilde{w}_m = w_m^{(0)} \cdot M_m$",
            fontsize=8.0, color='#c9d1d9', zorder=3)

    # 3.2 Iteration Loop Box
    draw_card(11.6, 3.5, 4.8, 8.4, "Iterative EM Optimization Loop", "Resolving the Chicken-and-Egg Diagnostic Prior", bg='#1c2128', border='#388bfd')

    # E-Step Box
    draw_card(11.8, 7.6, 4.4, 3.7, "E-Step: Latent Diagnosis Expectation", "Retrieve Diagnostic-Specific Prior", bg='#21262d', border='#a371f7', radius=0.15)
    ax.text(12.0, 10.4, "1. Fuse Representations with Current Weights:", fontsize=8.0, fontweight='bold', color='#d2a8ff', zorder=3)
    ax.text(12.0, 9.9, r"$\mathbf{z}_{\mathrm{fused}}^{(k)} = \sum_{m \in \mathrm{avail}} w_m^{(k)} \cdot \mathbf{z}_m$", fontsize=9.0, color='#ffffff', zorder=3)
    ax.text(12.0, 9.1, "2. Current Working Hypothesis Diagnosis:", fontsize=8.0, fontweight='bold', color='#d2a8ff', zorder=3)
    ax.text(12.0, 8.6, r"$\hat{y}^{(k)} = \mathrm{argmax}\left(\mathrm{softmax}(f(\mathbf{z}_{\mathrm{fused}}^{(k)}))\right)$", fontsize=9.0, color='#7ee787', zorder=3)
    ax.text(12.0, 8.0, r"3. Query Empirical Prior: $\pi_m(\hat{y}^{(k)}) = \mathbf{W}_{\mathrm{prior}}[\hat{y}^{(k)}, m]$", fontsize=8.0, color='#79c0ff', zorder=3)

    # M-Step Box
    draw_card(11.8, 3.9, 4.4, 3.4, "M-Step: Dynamic Weight Update", "Prior Modulation & Conflict Penalization", bg='#21262d', border='#39c5cf', radius=0.15)
    ax.text(12.0, 6.4, "Update Modality Importance Weights:", fontsize=8.0, fontweight='bold', color='#76e3ea', zorder=3)
    ax.text(12.0, 5.7, r"$\mathbf{w}^{(k+1)} = \mathrm{softmax}(\alpha f(Q,R,U,C) + \beta \mathbf{\pi}(\hat{y}^{(k)}) + \gamma \log(p_{\mathrm{conflict}}))$",
            fontsize=8.2, color='#ffffff', zorder=3)
    ax.text(12.0, 4.9, r"• $\beta$: Diagnosis-adaptive modality preference" + "\n" + r"• $\gamma$: Suppress discordant outlier modalities" + "\n" + r"• $f$: 4-input evidence projection MLP",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    # 3.3 Convergence Check
    draw_card(11.6, 0.9, 4.8, 2.3, "Convergence Verification", "Stopping Criterion or Max Iterations ($K=3$)", bg='#161b22', border='#3fb950')
    ax.text(11.8, 2.5, "Convergence Criteria:", fontsize=8.5, fontweight='bold', color='#7ee787', zorder=3)
    ax.text(11.8, 2.1, "$\\hat{y}^{(k+1)} == \\hat{y}^{(k)} \\quad \\text{AND} \\quad \\max_m |w_m^{(k+1)} - w_m^{(k)}| < \\tau = 0.05$", fontsize=8.2, color='#ffffff', zorder=3)
    ax.text(11.8, 1.3, "Result: Optimal Fused Manifold $\\mathbf{z}_{\\text{fused}}^*$, Weights $\\mathbf{w}^*$", fontsize=8.0, color='#7ee787', zorder=3)

    # Arrows inside Stage 3
    draw_arrow((14.0, 12.3), (14.0, 11.9), color='#d29922', lw=2.0)
    draw_arrow((14.0, 7.6), (14.0, 7.3), text="Prior Pi", color='#a371f7', lw=1.8)
    draw_arrow((14.0, 3.9), (14.0, 3.2), color='#39c5cf', lw=2.0)

    # Feedback Loop Arrow (M-Step back to E-step)
    draw_arrow((16.2, 5.3), (16.2, 9.4), text="Iterate: k -> k+1", color='#f0883e', rad=0.55, lw=2.0)

    # Arrow from Stage 2 to Stage 3
    draw_arrow((10.7, 4.0), (11.6, 13.2), text="Passed (Q, R, C)", color='#3fb950', rad=-0.25, lw=2.0)

    # ==========================================
    # STAGE 4: 4-LEVEL CONSENSUS & CONFORMAL UNCERTAINTY
    # ==========================================
    draw_card(17.0, 0.6, 5.1, 15.1, "STAGE 4: CONSENSUS & UNCERTAINTY", "4-Level Hierarchy & 95% Conformal Set",
              bg='#11151c', border='#a371f7', border_width=2.0)

    # 4.1 4-Level Consensus Hierarchy
    draw_card(17.3, 5.2, 4.5, 9.6, "4-Level Consensus Hierarchy", "Multi-Model Arbitration Decision Engine", bg='#161b22', border='#a371f7')

    # Level 1
    draw_card(17.5, 12.3, 4.1, 2.0, "LEVEL 1: STRONG CONSENSUS", "", bg='#1c2a1e', border='#3fb950', radius=0.15)
    ax.text(17.7, 13.6, r"• $\geq 3$ Models Agree in Diagnosis", fontsize=8.0, color='#7ee787', zorder=3)
    ax.text(17.7, 13.1, r"• Fused Confidence $\geq 0.80$ & $C \geq 0.30$", fontsize=8.0, color='#7ee787', zorder=3)
    ax.text(17.7, 12.6, "Action: Auto-Approve to Orchestrator", fontsize=8.0, fontweight='bold', color='#ffffff', zorder=3)

    # Level 2
    draw_card(17.5, 9.9, 4.1, 2.1, "LEVEL 2: MODERATE AGREEMENT", "", bg='#21262d', border='#388bfd', radius=0.15)
    ax.text(17.7, 11.2, "• 2 Models Agree, 1 Disagrees", fontsize=8.0, color='#79c0ff', zorder=3)
    ax.text(17.7, 10.7, "• Confidence in [0.50, 0.80)", fontsize=8.0, color='#79c0ff', zorder=3)
    ax.text(17.7, 10.2, "Action: Weighted Arbitration via (Q, R)", fontsize=8.0, fontweight='bold', color='#ffffff', zorder=3)

    # Level 3
    draw_card(17.5, 7.5, 4.1, 2.1, "LEVEL 3: HIGH UNCERTAINTY", "", bg='#2c2212', border='#d29922', radius=0.15)
    ax.text(17.7, 8.8, r"• Confidence $< 0.50$ OR Mean $Q < 0.30$", fontsize=8.0, color='#f2cc60', zorder=3)
    ax.text(17.7, 8.3, "• Disagreement Across Modalities", fontsize=8.0, color='#f2cc60', zorder=3)
    ax.text(17.7, 7.8, "Action: ACQUIRE Targeted Modality", fontsize=8.0, fontweight='bold', color='#e3b341', zorder=3)

    # Level 4
    draw_card(17.5, 5.4, 4.1, 1.8, "LEVEL 4: CRITICAL CONFLICT", "", bg='#2d181b', border='#f85149', radius=0.15)
    ax.text(17.7, 6.5, "• Diametric Modality Contradiction", fontsize=8.0, color='#ff7b72', zorder=3)
    ax.text(17.7, 6.0, "• High OOD Anomaly Score", fontsize=8.0, color='#ff7b72', zorder=3)
    ax.text(17.7, 5.5, "Action: ESCALATE to Dermatologist", fontsize=8.0, fontweight='bold', color='#ff7b72', zorder=3)

    # 4.2 Conformal Uncertainty Quantification
    draw_card(17.3, 0.9, 4.5, 4.0, "Conformal Prediction (95%)", "Finite-Sample Statistical Coverage Guarantee", bg='#161b22', border='#39c5cf')
    ax.text(17.5, 4.1, "Coverage Calibration:", fontsize=8.5, fontweight='bold', color='#76e3ea', zorder=3)
    ax.text(17.5, 3.6, r"$P(y \in \Gamma_{0.95}(x)) \geq 1 - \alpha = 0.95$", fontsize=9.0, color='#ffffff', zorder=3)
    ax.text(17.5, 2.9, "Non-Conformity Calibration Score:\n" + r"$s_i = 1 - p(y_i \mid x_i)$ on holdout calibration split", fontsize=7.8, color='#c9d1d9', zorder=3)
    ax.text(17.5, 2.0, "Prediction Set Output:\n" + r"$\Gamma_{0.95}(x) = \{c \in \mathcal{Y} : s(x, c) \leq \hat{q}_{1 - \alpha}\}$", fontsize=8.5, fontweight='bold', color='#7ee787', zorder=3)
    ax.text(17.5, 1.2, "Clinical Safety: Singleton = Confident, Multiple = Ambiguous", fontsize=7.5, color='#8b949e', zorder=3)

    # Arrow from Stage 3 to Stage 4
    draw_arrow((16.4, 2.0), (17.3, 10.0), text="z_fused*, w*", color='#7ee787', rad=-0.2, lw=2.0)

    # ==========================================
    # STAGE 5: ORCHESTRATOR & STAGE 2 VERIFICATION
    # ==========================================
    draw_card(22.4, 0.6, 5.0, 15.1, "STAGE 5: SYNTHESIS & VERIFICATION", "Orchestrator Agent & Clinical Output",
              bg='#11151c', border='#3fb950', border_width=2.0)

    # 5.1 Orchestrator Synthesis
    draw_card(22.7, 10.8, 4.4, 4.0, "1. Orchestrator Synthesis Agent", "Tool-Calling Multi-Agent Integrator", bg='#161b22', border='#388bfd')
    ax.text(22.9, 14.1, "Structured Evidence Ingestion:", fontsize=8.5, fontweight='bold', color='#79c0ff', zorder=3)
    ax.text(22.9, 13.6, "• Aggregates Vision Team inspection\n• Scribe extracted clinical facts\n• Evidence scores ($Q, R, C, S, M$)\n• Selected Consensus Level (1 to 4)",
            fontsize=8.0, color='#c9d1d9', zorder=3)
    ax.text(22.9, 11.5, "LLM Synthesis Generator:\nDeterministic Generation ($T = 0$)", fontsize=8.2, fontweight='bold', color='#7ee787', zorder=3)

    # 5.2 Stage 2 Verification (Hallucination Prevention)
    draw_card(22.7, 5.8, 4.4, 4.7, "2. Stage 2 Verification Layer", "Two-Tiered Clinical Hallucination Barrier", bg='#1c2128', border='#f85149')
    
    # Tier 1
    draw_card(22.9, 8.4, 4.0, 1.7, "Tier 1: Rule-Based Fact Audit", "", bg='#21262d', border='#d29922', radius=0.15)
    ax.text(23.1, 9.4, "• Every claim checked against JSON trace", fontsize=7.8, color='#f2cc60', zorder=3)
    ax.text(23.1, 8.9, "• Regex checks for fabricated symptoms", fontsize=7.8, color='#f2cc60', zorder=3)
    ax.text(23.1, 8.5, "Fail → Regenerate Conservative Report", fontsize=7.8, fontweight='bold', color='#ff7b72', zorder=3)

    # Tier 2
    draw_card(22.9, 6.2, 4.0, 1.9, "Tier 2: LLM Consistency Check", "", bg='#21262d', border='#a371f7', radius=0.15)
    ax.text(23.1, 7.4, "• Verification LLM inspects report draft", fontsize=7.8, color='#d2a8ff', zorder=3)
    ax.text(23.1, 6.9, "• Validates diagnostic reasoning logic", fontsize=7.8, color='#d2a8ff', zorder=3)
    ax.text(23.1, 6.5, "Fail → ESCALATE ('Verification Failed')", fontsize=7.8, fontweight='bold', color='#ff7b72', zorder=3)

    # 5.3 Final Clinical Output & Evidence Trace
    draw_card(22.7, 0.9, 4.4, 4.6, "3. Final Auditable Clinical Output", "Actionable Output & Complete Audit Trail", bg='#161b22', border='#3fb950')
    ax.text(22.9, 4.7, "Diagnostic Decisions:", fontsize=8.5, fontweight='bold', color='#7ee787', zorder=3)
    ax.text(22.9, 4.2, "• Primary Predicted Diagnosis: $\\hat{y} \\in \\mathcal{Y}$\n• Calibrated Confidence: $c = \\max P(y \\mid x)$\n• 95% Conformal Set: $\\Gamma_{0.95}(x)$\n• Decision Gate: PROCEED / ACQUIRE / ESCALATE",
            fontsize=8.0, color='#ffffff', zorder=3)
    ax.text(22.9, 2.5, "Auditable Evidence Trace:", fontsize=8.5, fontweight='bold', color='#79c0ff', zorder=3)
    ax.text(22.9, 2.0, "• Modality Weights $\\mathbf{w}^* = [w_{\\text{img}}, w_{\\text{text}}, w_{\\text{meta}}]$\n• Quality $\\mathbf{Q}$, Reliability $\\mathbf{R}$, Consistency $C$\n• Immutable Trace JSON saved to DB & WandB",
            fontsize=7.8, color='#c9d1d9', zorder=3)

    # Arrows in Stage 5
    draw_arrow((21.8, 12.0), (22.7, 12.8), text="Consensus L1/L2", color='#3fb950', lw=2.0)
    draw_arrow((24.9, 10.8), (24.9, 10.5), color='#388bfd', lw=2.0)
    draw_arrow((24.9, 5.8), (24.9, 5.5), text="Audit Passed", color='#3fb950', lw=2.0)

    # Acquisition Loop Back Arrow
    draw_arrow((19.5, 7.5), (3.1, 0.6), text="Level 3: ACQUIRE Evidence (Trigger Targeted Retest)",
               color='#d29922', rad=-0.35, lw=2.2, ls='--')

    # Escalation Exit Arrow
    draw_arrow((19.5, 5.4), (20.5, 0.2), text="Level 4 / OOD: ESCALATE (Human Dermatologist Panel)",
               color='#f85149', rad=0.2, lw=2.2, ls='--')

    # Footer note
    ax.text(14.0, 0.25, "DERMA-GUARD v4.0 • Designed for Safety Under Imperfect Multimodal Evidence • Conformal Guarantee $1-\\alpha = 0.95$ • Evaluated on 15-Condition DERMA-STRESS Benchmark",
            fontsize=9.0, color='#8b949e', ha='center', va='center', zorder=3)

    # Save outputs
    plt.savefig(output_png_path, format='png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(output_jpg_path, format='jpg', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved high-res architecture diagram to:\n  - {output_png_path}\n  - {output_jpg_path}")

if __name__ == "__main__":
    base_dir = r"d:\VIT BOOKS\PROJECT 1\Project"
    saved_models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    png_path = os.path.join(saved_models_dir, "derma_guard_architecture_workflow.png")
    jpg_path = os.path.join(saved_models_dir, "derma_guard_architecture_workflow.jpg")
    
    draw_architecture_diagram(png_path, jpg_path)
