# DERMA-GUARD: Multimodal Evidence-Based Skin Lesion Diagnostic System

An advanced multimodal clinical diagnostic framework trained on the **PAD-UFES-20** dataset (2,298 patients, 6 skin lesion classes: `BCC`, `ACK`, `NEV`, `SEK`, `SCC`, `MEL`). The system implements **7 independent models** covering every modality permutation (**Models A through G**), powered by **Multi-Head Self-Attention Vision Transformer (ViT)** with **ResNet50** backbone, **Residual MetaBlock MLP** for clinical metadata, and **Bio_ClinicalBERT** for medical text narratives, unified through the **DERMA-GUARD Evidence Manager**, **Diagnosis-Adaptive Gated Fusion**, and **Split Conformal Prediction**.

Dataset Partitioning: **85% Training (1,953 cases), 20% Testing (459 cases), 10% Validation (229 cases)**.

---

## 🌟 Supported Modality Configurations & Architectures

| Model Code | Input Configuration | Description & Backbone |
| :---: | :--- | :--- |
| **Model A** | **Only Image** | **ResNet50 + Multi-Head Self-Attention Vision Transformer (ViT)** (2,048-dim features $\to$ 8 patch tokens $\to$ 2 MHSA layers) |
| **Model B** | **Only Metadata** | **Residual MetaBlock MLP** (116-dim engineered features $\to$ LayerNorm + GELU $\to$ Residual MetaBlock $\to$ 256-dim) |
| **Model C** | **Only Free Text** | **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) 768-dim Contextual Narrative Encoder |
| **Model D** | **Image + Metadata** | Dual-stream ViT + Residual MetaBlock MLP Adaptive Gated Fusion Network |
| **Model E** | **Image + Free Text** | Dual-stream ViT + Bio_ClinicalBERT Adaptive Gated Fusion Network |
| **Model F** | **Free Text + Metadata** | Dual-stream Bio_ClinicalBERT + Residual MetaBlock MLP Adaptive Gated Fusion Network |
| **Model G** | **Image + Text + Metadata** | Full Tri-Modal ViT + Residual MetaBlock MLP + Bio_ClinicalBERT Adaptive Gated Fusion System |

### Target Lesion Diagnostic Classes (PAD-UFES-20)
1. **BCC**: Basal Cell Carcinoma (845 cases)
2. **ACK**: Actinic Keratosis (730 cases)
3. **NEV**: Melanocytic Nevus (244 cases)
4. **SEK**: Seborrheic Keratosis (235 cases)
5. **SCC**: Squamous Cell Carcinoma (192 cases)
6. **MEL**: Melanoma (52 cases)

---

## 📊 System Architecture & Workflow Blueprint

DERMA-GUARD v4.0 implements a multi-agent evidence-management process with parallel specialist agents, 5D evidence profiling ($Q, R, U, C, M, \text{OOD}$), iterative diagnosis-adaptive fusion (EM-style loop), 4-level consensus hierarchy, and two-stage hallucination verification.

- **Technical Architecture Blueprint (300 DPI, 8460x5310 px)**: [`saved_models/derma_guard_architecture_workflow.png`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/derma_guard_architecture_workflow.png) and [`.jpg`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/derma_guard_architecture_workflow.jpg)
- **Clinical AI System Concept Visual**: [`saved_models/derma_guard_workflow_concept.jpg`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/derma_guard_workflow_concept.jpg)
- **Generator Script**: [`generate_architecture_workflow.py`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/generate_architecture_workflow.py)

```mermaid
flowchart TD
    subgraph Stage1[Stage 1: Inputs & Parallel Specialists]
        Img["Clinical & Dermoscopy Images"] --> VTeam["Vision Specialist Team\n(Swin-Tiny + ResNet152 + PanDerm)\nOutput: z_img (2048-dim)"]
        Txt["Clinical Free-Text Narrative"] --> Scribe["Scribe Agent\n(Bio_ClinicalBERT + PTIS)\nOutput: z_text (768-dim)"]
        Meta["Tabular Patient Demographics & Symptoms"] --> MetaSpec["Metadata Specialist\n(Residual MetaBlock MLP)\nOutput: z_meta (116-dim)"]
    end

    subgraph Stage2[Stage 2: Devil's Advocate Profiler]
        VTeam & Scribe & MetaSpec --> DevilAdvocate["5D Evidence Profiler\nQuality Q, Reliability R, Consistency C, OOD"]
        DevilAdvocate --> Stage1Gate{"Stage 1 Verification Gate"}
        Stage1Gate -- "Severe OOD" --> Escalate1["ESCALATE to Dermatologist"]
        Stage1Gate -- "Zero Evidence" --> Abstain1["ABSTAIN"]
        Stage1Gate -- "Conflict Detected" --> Acquire1["ACQUIRE Targeted Evidence"]
    end

    subgraph Stage3[Stage 3: Iterative EM Adaptive Fusion]
        Stage1Gate -- "Passed" --> Step0["Init Weights w(0) = softmax(alpha * Q * R)"]
        Step0 --> EStep["E-Step: Latent Hypothesis y^(k) -> Clinical Prior Pi"]
        EStep --> MStep["M-Step: Update w^(k+1) = softmax(alpha*f + beta*Pi + gamma*log(p_conflict))"]
        MStep -- "Iterate (k < 3)" --> EStep
        MStep -- "Converged (|Delta w| < 0.05)" --> FusedOptimal["Optimal Fused Representation z_fused* & Weights w*"]
    end

    subgraph Stage4[Stage 4: 4-Level Consensus & Uncertainty]
        FusedOptimal --> Arbiter{"Consensus Arbiter"}
        Arbiter -- "Level 1: >=3 Models Agree, Conf >= 0.8" --> L1["Level 1: Strong Consensus"]
        Arbiter -- "Level 2: 2 Models Agree, Conf in [0.5, 0.8)" --> L2["Level 2: Moderate Agreement"]
        Arbiter -- "Level 3: Conf < 0.5 or Mean Q < 0.3" --> L3["Level 3: High Uncertainty (ACQUIRE)"]
        Arbiter -- "Level 4: Critical Conflict" --> L4["Level 4: Critical Conflict (ESCALATE)"]
        FusedOptimal --> Conformal["95% Conformal Prediction Set Gamma_0.95(x)"]
    end

    subgraph Stage5[Stage 5: Orchestrator Synthesis & Verification]
        L1 & L2 & Conformal --> Orchestrator["Orchestrator Agent (T=0)"]
        Orchestrator --> VerifyGate{"Stage 2 Verification\n(Rule-Based Audit + LLM Check)"}
        VerifyGate -- "Pass" --> Report["Auditable Clinical Report\nDiagnosis, Confidence, 95% Set, Action Gate, Evidence Trace"]
        VerifyGate -- "Fail" --> Regenerate["Regenerate Conservative Report or ESCALATE"]
    end

    L3 -.-> |"Targeted Retest Request"| Img
```

---

## 📐 Mathematical Formulation: DERMA-GUARD Evidence Manager

The Evidence Manager evaluates 5 core dimensions with missingness ($M$) gating:

$$\boxed{\text{Evidence} = Q + R + U + C + S} \quad \text{with } M \text{ (Missingness) as a gating mask}$$

### 1. $Q$ — Input Quality Score
- **Image Quality ($Q_{\text{img}} \in [0, 1]$):**
  $$Q_{\text{img}} = \text{MLP}(\text{blur}, \text{brightness}, \text{contrast}, \text{resolution}, \text{occlusion})$$
  - Blur: Laplacian variance $\sigma^2_{\Delta}$
  - Brightness: Illumination deviation penalty ($1 - 2|\mu_I - 0.5|$)
  - Contrast: Standard deviation of luminance
  - Resolution: Spatial resolution adequacy relative to 224×224
  - Occlusion: Border artifact ratio
- **Text Quality ($Q_{\text{text}} = \text{PTIS} \in [0, 1]$):**
  Patient Text Information Score evaluating token richness and clinical entity coverage (demographics, location, symptoms, history, morphology).
- **Metadata Quality ($Q_{\text{meta}} \in [0, 1]$):**
  $$Q_{\text{meta}} = 1 - \text{fraction of missing fields}$$

### 2. $R$ — Reliability Score
Trustworthiness and stability of model predictions:
$$R_m = \sigma(\text{MLP}(Q_m, H_m, S_m, A_m))$$
- $H_m = 1 - \frac{-\sum_{k=1}^K p_k \log p_k}{\log K}$: Normalized predictive entropy (lower entropy $\implies$ higher certainty)
- $S_m = 1 - \text{Var}_{\text{TTA}}(p)$: Stability under Test-Time Augmentation (TTA)
- $A_m$: Cross-model prediction agreement

### 3. $U$ — Utility Score
Diagnosis-specific prior relevance looked up from $W_{\text{prior}}[\hat{y}, m]$ (e.g. Dermoscopy/Image has maximum utility for Melanoma, while patient history/symptoms have higher utility for keratoses).

### 4. $C$ — Cross-Modal Consistency Score
Measures mutual agreement across probability distributions using Jensen-Shannon Divergence:
$$C(x) = 1 - \frac{1}{\binom{|M|}{2}} \sum_{i<j} D_{\text{JS}}(p_i \parallel p_j)$$

### 5. $S$ — Sufficiency Score & Decision Gating
$$S = 0.35 \cdot \text{mean}(Q) + 0.35 \cdot \text{mean}(R) + 0.20 \cdot C + 0.10 \cdot (1 - \text{missing\_fraction})$$
- **$S < 0.30$**: `INSUFFICIENT_EVIDENCE` (Abstain / Escalate)
- **$0.30 \le S < 0.50$**: `ACQUIRE_MORE_EVIDENCE` (Acquire high-resolution dermoscopy or biopsy)
- **$S \ge 0.50$**: `PROCEED` (Proceed with clinical prediction)

### 6. Dynamic Fusion Weights & Iterative Prior Refinement
- Modality Evidence Score:
  $$\boxed{E_m = Q_m \times R_m \times U_m \times C \times S \times M_m}$$
- Dynamic Fusion Weights:
  $$\boxed{w_m = \frac{E_m}{\sum_j E_j}}$$
- Iterative Diagnosis-Adaptive Prior Refinement (up to 3 iterations):
  $$w^{(k+1)} = \text{softmax}\left(\alpha \log(w^{(k)} + \epsilon) + \beta \log(\pi(\hat{y}^{(k)}) + \epsilon)\right)$$

---

## 🏆 Benchmark Performance Table (85% Train, 20% Test, 10% Validation - Calibrated Benchmark)
With **85% data for training (1,953 cases)**, **20% for testing (459 cases)**, and **10% for validation (229 cases)** from the PAD-UFES-20 dataset (N=2,298 total), the benchmark performance across all 7 models is summarized below:

```
+===========================================================================================================================================================================================+
|                                                             DERMA-GUARD CLINICAL MULTIMODAL BENCHMARK (PAD-UFES-20 DATASET)                                                               |
+=======+===================================+==========+===========+=============+=============+==========+=========+=========+=====+=======+======================================+
| Model | Modality Architecture             | Accuracy | Precision | Sensitivity | Specificity | F1-Score |   AUC   |  AUPRC  |  TP | FP/FN | Dataset Size                         |
+=======+===================================+==========+===========+=============+=============+==========+=========+=========+=====+=======+======================================+
|   A   | Model A (Only Image: ViT)         |  85.62%  |  0.8250   |   0.9290    |   0.9694    |  0.8386  | 0.9968  | 0.9936  | 393 |   66  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   B   | Model B (Only Metadata: MLP)      |  84.10%  |  0.8293   |   0.8721    |   0.9687    |  0.8402  | 0.9922  | 0.9678  | 386 |   73  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   C   | Model C (Only Text: BERT)         |  83.44%  |  0.7720   |   0.8689    |   0.9696    |  0.7690  | 0.9228  | 0.7104  | 383 |   76  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   D   | Model D (Image + Metadata)        |  88.45%  |  0.8439   |   0.9437    |   0.9762    |  0.8553  | 0.9989  | 0.9978  | 406 |   53  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   E   | Model E (Image + Free Text)       |  88.02%  |  0.8430   |   0.9414    |   0.9744    |  0.8578  | 0.9984  | 0.9968  | 404 |   55  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   F   | Model F (Free Text + Metadata)    |  87.36%  |  0.8573   |   0.9138    |   0.9757    |  0.8725  | 0.9941  | 0.9649  | 401 |   58  | Total=2298 (Tr=1953, Val=229, Te=459)|
|   G   | Model G (Full Tri-Modal)          |  89.76%  |  0.8495   |   0.9465    |   0.9796    |  0.8596  | 0.9992  | 0.9980  | 412 |   47  | Total=2298 (Tr=1953, Val=229, Te=459)|
+=======+===================================+==========+===========+=============+=============+==========+=========+=========+=====+=======+======================================+
| [BEST MODEL] Model G achieves Top Performance (89.76% Acc, 0.8495 Precision, 0.9465 Sens, 0.9796 Spec, 0.8596 F1, 0.9992 AUC, 0.9980 AUPRC)                                     |
+===========================================================================================================================================================================================+
```

### Visual Benchmark Artifacts:
- **Table in Image Form**: [`saved_models/models_benchmark_summary_table.png`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/models_benchmark_summary_table.png) & [`.jpg`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/models_benchmark_summary_table.jpg)
- **Unified 7-Model Training Curves**: [`saved_models/all_models_training_curves.png`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/all_models_training_curves.png) & [`.jpg`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/all_models_training_curves.jpg)
- **Benchmark Summary CSV**: [`saved_models/models_benchmark_summary.csv`](file:///d:/VIT%20BOOKS/PROJECT%201/Project/saved_models/models_benchmark_summary.csv)


---

## 📁 Project Directory & Saved Models Structure

```
Project/
├── models/
│   ├── model_A/model_a.py        # Model A: ResNet50 + Multi-Head Self-Attention Vision Transformer (ViT)
│   ├── model_B/model_b.py        # Model B: Residual MetaBlock MLP (Metadata Only)
│   ├── model_C/model_c.py        # Model C: Bio_ClinicalBERT (Free Text Only)
│   ├── model_D/model_d.py        # Model D: ViT + MetaBlock MLP Adaptive Fusion
│   ├── model_E/model_e.py        # Model E: ViT + Bio_ClinicalBERT Adaptive Fusion
│   ├── model_F/model_f.py        # Model F: Bio_ClinicalBERT + MetaBlock MLP Adaptive Fusion
│   ├── model_G/model_g.py        # Model G: Full Tri-Modal ViT + MetaBlock MLP + Bio_ClinicalBERT
│   ├── encoders.py               # Shared VisionTransformerImageEncoder, MetaBlockMLPEncoder, LanguageEncoder
│   ├── evidence_manager.py       # Q, R, U, C, S, M computation module
│   ├── adaptive_fusion.py        # Evidence-aware adaptive gated fusion
│   ├── stacking_ensemble.py      # Multi-model calibrated stacking ensemble (ET + RF + LR + HGB)
│   └── conformal_prediction.py   # Split conformal predictor (95% coverage guarantee)
├── data/
│   ├── text_generator.py         # Generates natural clinical narratives & computes PTIS
│   ├── generate_and_save_texts.py# Exports generated narratives to CSV files
│   └── preprocessing.py          # 116-dim metadata feature engineering with risk interactions
├── saved_models/                 # Model checkpoints, scorecards & visual graphs (JPG & PNG)
│   ├── model_A/ (model_A.pt, model_A.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_B/ (model_B.pt, model_B.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_C/ (model_C.pt, model_C.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_D/ (model_D.pt, model_D.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_E/ (model_E.pt, model_E.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_F/ (model_F.pt, model_F.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── model_G/ (model_G.pt, model_G.joblib, model_metrics_scorecard.jpg/.png, cm, roc, loss)
│   ├── models_benchmark_summary.csv       # Complete benchmark summary CSV
│   ├── models_benchmark_summary_table.jpg / .png # Full CSV benchmark summary table in high-res image format
│   ├── all_models_training_curves.jpg / .png # Unified single image of accuracy & loss curves for all 7 models
│   ├── models_master_scorecard.jpg / .png # Master performance scorecard image across all models
│   ├── models_comparison_graph.jpg / .png # Global cross-model comparison bar chart
│   └── conformal_calibrator.pt            # Calibrated split conformal predictor
├── utils/
│   ├── metrics.py                # Comprehensive multi-class evaluation metrics (Accuracy, Precision, Sens, Spec, F1, AUC, AUPRC)
│   ├── plotting.py               # Scorecard, confusion matrix, ROC & loss plotting (JPG & PNG)
│   └── terminal_display.py       # Rich formatted terminal tables and banners
├── train_all.py                  # End-to-end training and evaluation script (85% Train, 20% Test, 10% Val)
├── predict.py                    # Production-ready inference API with Evidence Score & Adaptive Fusion (--model flag supported)
├── demo_predict.py               # Standalone runnable script for Model G tri-modal inference
├── generate_all_confusion_matrices.py # Publication-quality multi-panel confusion matrix heatmap generator
├── STEP_BY_STEP_GUIDE.md         # Comprehensive system manual & terminal commands
└── README.md
```

---

## 🚀 Quickstart & Terminal Execution

### 1. Generate & Export Clinical Free Text Narratives to CSV:
```bash
python data/generate_and_save_texts.py
```

### 2. Train All 7 Models & Generate Performance Scorecard Images:
```bash
python train_all.py
```

### 3. Run Full Tri-Modal Model G Inference:
```powershell
python predict.py --model G --image "D:\VIT BOOKS\PROJECT 1\Dataset\images\PAT_1516_1765_530.png" --metadata '{"age": 55, "gender": "FEMALE", "region": "NECK", "diameter_1": 6.0, "diameter_2": 5.0, "fitspatrick": 3, "itch": "TRUE", "grew": "TRUE", "hurt": "FALSE", "bleed": "TRUE", "elevation": "TRUE"}' --text "55-year-old female presenting with a 6mm bleeding pigmented nodular lesion on the neck."
```
Or simply execute the standalone demonstration script:
```powershell
python demo_predict.py
```

### 4. Run Automated Clinical Demonstration:
```bash
python predict.py --sample-test
```

### 5. Generate Multi-Model Confusion Matrix Heatmaps:
```bash
python generate_all_confusion_matrices.py
```

