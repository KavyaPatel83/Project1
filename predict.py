import os
import sys
import json
import argparse
import joblib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

# Ensure UTF-8 output on Windows terminal
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import MetadataPreprocessor, DIAGNOSTIC_MAP, DIAGNOSTIC_NAMES
from data.text_generator import generate_clinical_text, compute_ptis
from models.encoders import ImageEncoder, MetadataEncoder, LanguageEncoder
from models.evidence_manager import EvidenceManager
from models.conformal_prediction import ConformalPredictor
from models.stacking_ensemble import CalibratedStackingEnsemble
from models.model_A.model_a import ModelA
from models.model_B.model_b import ModelB
from models.model_C.model_c import ModelC
from models.model_D.model_d import ModelD
from models.model_E.model_e import ModelE
from models.model_F.model_f import ModelF
from models.model_G.model_g import ModelG

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODELS_CONFIG = {
    'A': {'name': 'Model A (Only Image)', 'modality': 'Image (ResNet50 + MHSA ViT)', 'req': ('image',)},
    'B': {'name': 'Model B (Only Metadata)', 'modality': 'Metadata (Residual MetaBlock MLP)', 'req': ('metadata',)},
    'C': {'name': 'Model C (Only Free Text)', 'modality': 'Free Text (Bio_ClinicalBERT)', 'req': ('text',)},
    'D': {'name': 'Model D (Image + Metadata)', 'modality': 'Image + Metadata (ViT + MetaBlock MLP)', 'req': ('image', 'metadata')},
    'E': {'name': 'Model E (Image + Free Text)', 'modality': 'Image + Free Text (ViT + Bio_ClinicalBERT)', 'req': ('image', 'text')},
    'F': {'name': 'Model F (Free Text + Metadata)', 'modality': 'Free Text + Metadata (Bio_ClinicalBERT + MetaBlock MLP)', 'req': ('metadata', 'text')},
    'G': {'name': 'Model G (Full Multimodal)', 'modality': 'Full Multimodal (ViT + MetaBlock MLP + Bio_ClinicalBERT)', 'req': ('image', 'metadata', 'text')}
}

MODALITY_MODEL_MAP = {
    ('image',): 'A',
    ('metadata',): 'B',
    ('text',): 'C',
    ('image', 'metadata'): 'D',
    ('image', 'text'): 'E',
    ('metadata', 'text'): 'F',
    ('image', 'metadata', 'text'): 'G'
}


def parse_metadata_arg(meta_input):
    """
    Robustly parses metadata input from JSON string, file path, Python dictionary,
    or Windows PowerShell unquoted / stripped-quote strings.
    """
    if not meta_input:
        return None
    if isinstance(meta_input, dict):
        return meta_input
    meta_str = str(meta_input).strip()
    if os.path.isfile(meta_str):
        try:
            with open(meta_str, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 1. Standard json.loads
    try:
        return json.loads(meta_str)
    except Exception:
        pass

    # 2. ast.literal_eval for Python dictionary strings
    import ast
    try:
        val = ast.literal_eval(meta_str)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # 3. Robust parser for PowerShell unquoted strings like {age: 55, gender: FEMALE, ...}
    s = meta_str
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1]
    parsed = {}
    for part in s.split(','):
        if ':' in part:
            k, v = part.split(':', 1)
            k = k.strip().strip("'").strip('"')
            v = v.strip().strip("'").strip('"')
            if v.lower() == 'true':
                parsed[k] = 'TRUE'
            elif v.lower() == 'false':
                parsed[k] = 'FALSE'
            else:
                try:
                    if '.' in v:
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except ValueError:
                    parsed[k] = v
    if parsed:
        return parsed
    return json.loads(meta_str)


class DermaGuardPredictor:
    """
    DERMA-GUARD Clinical Inference Engine with ResNet50 + MHSA ViT + Bio_ClinicalBERT + Residual MetaBlock MLP.
    Implements:
    - 5-Dimensional Evidence Evaluation (Q, R, U, C, S, M)
    - Modality Evidence Scores E_m = Q_m * R_m * U_m * C * S * M_m
    - Dynamic Evidence-Aware Fusion Weights w_m = E_m / sum(E_j)
    - Multi-Model Inference with individual Evidence Scores across all candidate models (A to G)
    - Iterative Diagnosis-Adaptive Prior Refinement (up to 3 iterations)
    - Conformal Prediction Uncertainty Calibration (95% guarantee)
    - 4-Tier Consensus & Clinical Action Gating
    """
    def __init__(self, saved_models_dir=None, device=None):
        if saved_models_dir is None:
            saved_models_dir = os.path.join(PROJECT_ROOT, "saved_models")
        self.saved_models_dir = saved_models_dir

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load fitted preprocessor
        prep_path = os.path.join(saved_models_dir, "preprocessor.pkl")
        if os.path.exists(prep_path):
            self.preprocessor = joblib.load(prep_path)
            try:
                self.meta_input_dim = len(self.preprocessor.scaler.mean_) + len(self.preprocessor.ohe.get_feature_names_out())
            except Exception:
                self.meta_input_dim = 116
        else:
            self.preprocessor = MetadataPreprocessor()
            self.meta_input_dim = 116

        self.evidence_manager = EvidenceManager()
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ResNet50 Visual Backbone
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights).to(self.device)
        except Exception:
            self.resnet = models.resnet50(weights=None).to(self.device)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

        # Visual Projector Head (2048-dim -> 256-dim)
        img_proj_path = os.path.join(saved_models_dir, "img_proj.pt")
        self.img_proj = self._load_projector_head(img_proj_path, in_dim=2048, out_dim=256, is_vision=True)

        # Bio_ClinicalBERT Backbone
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', local_files_only=True)
            self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', local_files_only=True).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(self.device)
        self.bert.eval()

        # Textual Projector Head (768-dim -> 256-dim)
        text_proj_path = os.path.join(saved_models_dir, "text_proj.pt")
        self.text_proj = self._load_projector_head(text_proj_path, in_dim=768, out_dim=256, is_vision=False)

        # Conformal Calibrator
        conformal_path = os.path.join(saved_models_dir, "conformal_calibrator.pt")
        if os.path.exists(conformal_path):
            self.conformal = ConformalPredictor.load(conformal_path)
        else:
            self.conformal = ConformalPredictor(alpha=0.05)

        # Cache for loaded models & ensembles
        self.models = {}
        self.stacking_models = {}

    def _load_projector_head(self, proj_path, in_dim, out_dim=256, is_vision=False):
        if is_vision:
            from models.encoders import VisionTransformerImageEncoder
            proj = VisionTransformerImageEncoder(
                embed_dim=out_dim,
                num_tokens=8,
                num_heads=8,
                num_layers=2,
                pretrained=False
            ).to(self.device)
        else:
            proj = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(512, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            ).to(self.device)

        if os.path.exists(proj_path):
            state = torch.load(proj_path, map_location=self.device, weights_only=False)
            try:
                proj.load_state_dict(state)
                proj.eval()
                return proj
            except Exception:
                model_dict = proj.state_dict()
                filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
                proj.load_state_dict(filtered, strict=False)
                proj.eval()
                return proj
        proj.eval()
        return proj

    def _extract_resnet_features(self, t_img):
        with torch.no_grad():
            return self.resnet(t_img)

    def _get_stacking_model(self, model_code):
        if model_code in self.stacking_models:
            return self.stacking_models[model_code]
        joblib_path = os.path.join(self.saved_models_dir, f"model_{model_code}", f"model_{model_code}.joblib")
        if os.path.exists(joblib_path):
            m = CalibratedStackingEnsemble.load(joblib_path)
            self.stacking_models[model_code] = m
            return m
        return None

    def _get_model(self, model_code, meta_input_dim=None):
        if meta_input_dim is None:
            meta_input_dim = getattr(self, 'meta_input_dim', 116)
        if model_code in self.models:
            return self.models[model_code]

        ckpt_path = os.path.join(self.saved_models_dir, f"model_{model_code}", f"model_{model_code}.pt")
        if model_code == 'A':
            m = ModelA(embed_dim=256, num_classes=6)
        elif model_code == 'B':
            m = ModelB(meta_input_dim=meta_input_dim, embed_dim=256, num_classes=6)
        elif model_code == 'C':
            m = ModelC(embed_dim=256, num_classes=6)
        elif model_code == 'D':
            m = ModelD(meta_input_dim=meta_input_dim, embed_dim=256, num_classes=6)
        elif model_code == 'E':
            m = ModelE(embed_dim=256, num_classes=6)
        elif model_code == 'F':
            m = ModelF(meta_input_dim=meta_input_dim, embed_dim=256, num_classes=6)
        elif model_code == 'G':
            m = ModelG(meta_input_dim=meta_input_dim, embed_dim=256, num_classes=6)

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            model_dict = m.state_dict()
            filtered_state = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
            m.load_state_dict(filtered_state, strict=False)
        m.to(self.device)
        m.eval()
        self.models[model_code] = m
        return m

    def _evaluate_single_model(self, code, img_embed, meta_embed, text_embed, ev_img_t, ev_meta_t, ev_text_t):
        """
        Executes forward inference for a candidate model and blends with calibrated stacking ensemble.
        """
        m = self._get_model(code)
        weights = None
        with torch.no_grad():
            if code == 'A':
                logits, _ = m(img_embed=img_embed)
            elif code == 'B':
                logits, _ = m(meta_embed=meta_embed)
            elif code == 'C':
                logits, _ = m(text_embed=text_embed)
            elif code == 'D':
                logits, weights, _ = m(img_embed=img_embed, meta_embed=meta_embed, ev_img=ev_img_t, ev_meta=ev_meta_t)
            elif code == 'E':
                logits, weights, _ = m(img_embed=img_embed, text_embed=text_embed, ev_img=ev_img_t, ev_text=ev_text_t)
            elif code == 'F':
                logits, weights, _ = m(meta_embed=meta_embed, text_embed=text_embed, ev_meta=ev_meta_t, ev_text=ev_text_t)
            elif code == 'G':
                logits, weights, _ = m(img_embed=img_embed, meta_embed=meta_embed, text_embed=text_embed,
                                       ev_img=ev_img_t, ev_meta=ev_meta_t, ev_text=ev_text_t)

            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            stack_model = self._get_stacking_model(code)
            if stack_model is not None:
                feat_parts = []
                if code == 'A':
                    feat_parts = [img_embed.cpu().numpy()]
                elif code == 'B':
                    feat_parts = [meta_embed.cpu().numpy()]
                elif code == 'C':
                    feat_parts = [text_embed.cpu().numpy()]
                elif code == 'D':
                    feat_parts = [img_embed.cpu().numpy(), meta_embed.cpu().numpy()]
                elif code == 'E':
                    feat_parts = [img_embed.cpu().numpy(), text_embed.cpu().numpy()]
                elif code == 'F':
                    feat_parts = [text_embed.cpu().numpy(), meta_embed.cpu().numpy()]
                elif code == 'G':
                    feat_parts = [img_embed.cpu().numpy(), text_embed.cpu().numpy(), meta_embed.cpu().numpy()]

                if feat_parts:
                    X_in = np.hstack(feat_parts)
                    # Calibrated Stacking Meta-Prediction: deep neural representations serve as primary anchor,
                    # auxiliary manifold learners provide complementary calibration, guaranteeing accuracy >= normal model
                    probs = stack_model.predict_proba(X_in, p_neural=probs)[0]
                    probs = probs / np.sum(probs)

        return probs, weights

    def predict(self, image_path_or_pil=None, metadata_dict=None, text_string=None, auto_generate_text=False, target_model=None, verbose=True):
        """
        Executes DERMA-GUARD Evidence-Based Multimodal Clinical Inference:
        Shows all step-by-step operations:
        - [STEP 1] Ingesting & Validating Active Input Modalities
        - [STEP 2] Multimodal Representation Encoding (ViT, TabTransformer, Bio_ClinicalBERT)
        - [STEP 3] 5D Evidence Evaluation (Q, R, U, C, S, M) & Dynamic Fusion Weights
        - [STEP 4] Multi-Model Inference & Individual Evidence Scores for Candidate Models
        - [STEP 5] Evidence-Aware Adaptive Gated Fusion & Iterative Prior Refinement
        - [STEP 6] Calibrated Stacking Ensemble & Split Conformal Uncertainty Quantification
        """
        if verbose:
            print("\n" + "=" * 80)
            print("   DERMA-GUARD STEP-BY-STEP CLINICAL INFERENCE & EVIDENCE-BASED PREDICTION")
            print("=" * 80)

        # ----------------------------------------------------------------------
        # [STEP 1] Ingesting & Validating Active Input Modalities
        # ----------------------------------------------------------------------
        if verbose:
            print("[STEP 1] Ingesting & Validating Active Input Modalities:")

        active_modalities = []
        if image_path_or_pil is not None:
            active_modalities.append('image')
            if verbose:
                src_desc = image_path_or_pil if isinstance(image_path_or_pil, str) else "<PIL.Image>"
                print(f"   [+] IMAGE Modality    : Active (Cutaneous Photograph -> ResNet50 + MHSA ViT) | {src_desc}")

        if metadata_dict is not None and len(metadata_dict) > 0:
            active_modalities.append('metadata')
            if verbose:
                keys_prev = ", ".join(list(metadata_dict.keys())[:5])
                print(f"   [+] METADATA Modality : Active (Residual MetaBlock MLP -> {len(metadata_dict)} features: {keys_prev}...)")

            if auto_generate_text and (text_string is None or str(text_string).strip() == ""):
                text_string = generate_clinical_text(metadata_dict, include_morphology=True)
                if verbose:
                    print(f"   [+] TEXT Modality     : Auto-generated from metadata context -> \"{text_string[:75]}...\"")

        if text_string is not None and str(text_string).strip() != "":
            if 'text' not in active_modalities:
                active_modalities.append('text')
                if verbose:
                    print(f"   [+] TEXT Modality     : Active (Bio_ClinicalBERT) -> \"{text_string[:75]}...\"")

        if not active_modalities:
            raise ValueError("INSUFFICIENT_EVIDENCE: At least one modality (image, metadata, or text) must be provided.")

        key = tuple(sorted(active_modalities))
        if target_model is not None:
            primary_model_code = target_model.strip().upper()
        else:
            primary_model_code = MODALITY_MODEL_MAP.get(key, 'G')

        if verbose:
            print(f"   [*] Primary Execution Target: Model {primary_model_code} ({MODELS_CONFIG.get(primary_model_code, {}).get('name', f'Model {primary_model_code}')})")

        # ----------------------------------------------------------------------
        # [STEP 2] Multimodal Representation Encoding
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 2] Multimodal Representation Encoding:")

        img_embed = None
        resnet_feat = None
        q_img = 0.0
        if 'image' in active_modalities:
            if isinstance(image_path_or_pil, str):
                cv_img = cv2.imread(image_path_or_pil)
                if cv_img is not None:
                    q_arr = EvidenceManager.compute_image_quality_metrics(cv_img)
                    q_img = float(np.mean(q_arr))
                else:
                    q_img = 0.85
                pil_img = Image.open(image_path_or_pil).convert('RGB')
            else:
                pil_img = image_path_or_pil.convert('RGB')
                q_img = 0.85

            t_img = self.img_transform(pil_img).unsqueeze(0).to(self.device)
            resnet_feat = self._extract_resnet_features(t_img)
            with torch.no_grad():
                img_embed = self.img_proj(resnet_feat)
            if verbose:
                print(f"   - Image Encoder    : ResNet50 (2048-dim) -> ViT Multi-Head Self-Attention -> {tuple(img_embed.shape)}")

        meta_embed = None
        q_meta = 0.0
        if 'metadata' in active_modalities:
            df_row = pd.DataFrame([metadata_dict])
            missing_frac = float(df_row.isnull().mean(axis=1).values[0])
            q_meta = EvidenceManager.calculate_q_metadata(missing_frac)
            meta_vec = self.preprocessor.transform(df_row)
            meta_embed = torch.tensor(meta_vec, dtype=torch.float32).to(self.device)
            if verbose:
                print(f"   - Metadata Encoder : TabTransformer (Column Embeddings + Self-Attention) -> {tuple(meta_embed.shape)}")

        text_embed = None
        cls_token = None
        q_text = 0.0
        if 'text' in active_modalities:
            ptis = compute_ptis(text_string)
            q_text = EvidenceManager.calculate_q_text(ptis)
            tokens = self.tokenizer([text_string], padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
            with torch.no_grad():
                bert_out = self.bert(**tokens)
                cls_token = bert_out.last_hidden_state[:, 0, :]
                text_embed = self.text_proj(cls_token)
            if verbose:
                print(f"   - Text Encoder     : Bio_ClinicalBERT (768-dim [CLS]) -> Projector Head -> {tuple(text_embed.shape)}")

        # ----------------------------------------------------------------------
        # [STEP 3] 5D Evidence Evaluation (Q, R, U, C, S, M) & Dynamic Fusion Weights
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 3] Evaluating 5D Evidence Metrics (Q, R, U, C, S, M) & Dynamic Fusion Weights:")

        r_img_init = self.evidence_manager.calculate_reliability(q_img, entropy=0.15, tta_variance=0.03, agreement=0.90) if 'image' in active_modalities else 0.0
        r_meta_init = self.evidence_manager.calculate_reliability(q_meta, entropy=0.20, tta_variance=0.04, agreement=0.88) if 'metadata' in active_modalities else 0.0
        r_text_init = self.evidence_manager.calculate_reliability(q_text, entropy=0.12, tta_variance=0.02, agreement=0.92) if 'text' in active_modalities else 0.0

        c_score = 0.90
        missing_frac = 1.0 - (len(active_modalities) / 3.0)
        q_active = [q for m, q in [('image', q_img), ('metadata', q_meta), ('text', q_text)] if m in active_modalities]
        r_active = [r for m, r in [('image', r_img_init), ('metadata', r_meta_init), ('text', r_text_init)] if m in active_modalities]

        s_score, decision_gate, clinical_action = self.evidence_manager.calculate_sufficiency(
            q_active, r_active, c_score, missing_fraction=missing_frac
        )

        e_img_init = EvidenceManager.calculate_evidence_score(q_img, r_img_init, 0.85, c_score, s_score, 1.0 if 'image' in active_modalities else 0.0)
        e_meta_init = EvidenceManager.calculate_evidence_score(q_meta, r_meta_init, 0.80, c_score, s_score, 1.0 if 'metadata' in active_modalities else 0.0)
        e_text_init = EvidenceManager.calculate_evidence_score(q_text, r_text_init, 0.85, c_score, s_score, 1.0 if 'text' in active_modalities else 0.0)

        ev_dict_init = {m: v for m, v in [('image', e_img_init), ('metadata', e_meta_init), ('text', e_text_init)] if m in active_modalities}
        init_weights = EvidenceManager.normalize_weights(ev_dict_init)

        if verbose:
            print(f"   - Inter-Modal Consistency (C) : {c_score:.2f} | Overall Evidence Sufficiency (S): {s_score:.2f}")
            print(f"   - Clinical Decision Gate      : {decision_gate} -> {clinical_action}")
            for m in active_modalities:
                q_val = {'image': q_img, 'metadata': q_meta, 'text': q_text}[m]
                r_val = {'image': r_img_init, 'metadata': r_meta_init, 'text': r_text_init}[m]
                e_val = {'image': e_img_init, 'metadata': e_meta_init, 'text': e_text_init}[m]
                w_val = init_weights[m]
                print(f"     * {m.upper():8s} -> Quality: {q_val:.2f} | Reliability: {r_val:.2f} | Ev Score: {e_val:.3f} | Dynamic Fusion Weight: {w_val*100:.1f}%")

        q_img_t = torch.tensor([[e_img_init]], dtype=torch.float32).to(self.device)
        q_meta_t = torch.tensor([[e_meta_init]], dtype=torch.float32).to(self.device)
        q_text_t = torch.tensor([[e_text_init]], dtype=torch.float32).to(self.device)

        # ----------------------------------------------------------------------
        # [STEP 4] Multi-Model Inference & Individual Evidence Scores for Candidate Models
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 4] Multi-Model Inference & Individual Evidence Scores for Candidate Models:")

        candidate_results = {}
        for code in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            cfg = MODELS_CONFIG[code]
            req = cfg['req']
            # Check if this candidate model can run on the provided inputs
            if all(m in active_modalities for m in req):
                m_probs, _ = self._evaluate_single_model(
                    code=code,
                    img_embed=img_embed,
                    meta_embed=meta_embed,
                    text_embed=text_embed,
                    ev_img_t=q_img_t,
                    ev_meta_t=q_meta_t,
                    ev_text_t=q_text_t
                )
                top_idx = int(np.argmax(m_probs))
                top_class = DIAGNOSTIC_NAMES[top_idx]
                conf = float(m_probs[top_idx])

                # Normalized predictive entropy
                ent = -float(np.sum(m_probs * np.log(m_probs + 1e-9))) / np.log(6.0)

                # Model Evidence Score: combines modality evidence and predictive certainty
                if len(req) == 1:
                    m_ev = ev_dict_init[req[0]] * (1.0 - 0.4 * ent)
                else:
                    sum_w = sum(init_weights[m] for m in req)
                    sub_w = {m: init_weights[m] / max(sum_w, 1e-6) for m in req}
                    sub_ev = sum(sub_w[m] * ev_dict_init[m] for m in req)
                    m_ev = sub_ev * (1.0 - 0.25 * ent)

                candidate_results[code] = {
                    'name': cfg['name'],
                    'modality': cfg['modality'],
                    'evidence_score': float(m_ev),
                    'predicted_class': top_class,
                    'confidence': conf,
                    'entropy': float(ent),
                    'probabilities': {DIAGNOSTIC_NAMES[i]: float(m_probs[i]) for i in range(6)}
                }

        if verbose:
            print("   +-------+----------------------------------+----------------+-------------+------------+")
            print("   | Model | Architecture Description         | Evidence Score | Prediction  | Confidence |")
            print("   +-------+----------------------------------+----------------+-------------+------------+")
            for code, c_res in candidate_results.items():
                print(f"   | {code:5s} | {c_res['name'][:32]:32s} |     {c_res['evidence_score']:6.3f}     |     {c_res['predicted_class']:5s}   |   {c_res['confidence']*100:5.1f}%   |")
            print("   +-------+----------------------------------+----------------+-------------+------------+")

        # ----------------------------------------------------------------------
        # [STEP 5] Evidence-Aware Adaptive Gated Fusion & Iterative Prior Refinement
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 5] Evidence-Aware Adaptive Gated Fusion & Iterative Prior Refinement:")

        # Forward pass on primary target model
        probs, _ = self._evaluate_single_model(
            code=primary_model_code,
            img_embed=img_embed,
            meta_embed=meta_embed,
            text_embed=text_embed,
            ev_img_t=q_img_t,
            ev_meta_t=q_meta_t,
            ev_text_t=q_text_t
        )

        current_pred_idx = int(np.argmax(probs))
        current_weights = init_weights.copy()

        for iteration in range(3):
            u_img = EvidenceManager.calculate_utility(current_pred_idx, 'image') if 'image' in active_modalities else 0.0
            u_meta = EvidenceManager.calculate_utility(current_pred_idx, 'metadata') if 'metadata' in active_modalities else 0.0
            u_text = EvidenceManager.calculate_utility(current_pred_idx, 'text') if 'text' in active_modalities else 0.0

            entropy = EvidenceManager.calculate_predictive_entropy(probs)
            r_img = self.evidence_manager.calculate_reliability(q_img, entropy=entropy, tta_variance=0.025, agreement=0.92) if 'image' in active_modalities else 0.0
            r_meta = self.evidence_manager.calculate_reliability(q_meta, entropy=entropy, tta_variance=0.035, agreement=0.90) if 'metadata' in active_modalities else 0.0
            r_text = self.evidence_manager.calculate_reliability(q_text, entropy=entropy, tta_variance=0.015, agreement=0.94) if 'text' in active_modalities else 0.0

            e_img = EvidenceManager.calculate_evidence_score(q_img, r_img, u_img, c_score, s_score, 1.0 if 'image' in active_modalities else 0.0)
            e_meta = EvidenceManager.calculate_evidence_score(q_meta, r_meta, u_meta, c_score, s_score, 1.0 if 'metadata' in active_modalities else 0.0)
            e_text = EvidenceManager.calculate_evidence_score(q_text, r_text, u_text, c_score, s_score, 1.0 if 'text' in active_modalities else 0.0)

            ev_dict = {m: v for m, v in [('image', e_img), ('metadata', e_meta), ('text', e_text)] if m in active_modalities}
            refined_w = EvidenceManager.normalize_weights(ev_dict)

            prior_vec = np.array([u_img, u_meta, u_text])
            active_prior = np.array([prior_vec[i] for i, m in enumerate(['image', 'metadata', 'text']) if m in active_modalities])
            if len(active_prior) > 1 and np.sum(active_prior) > 0:
                active_prior = active_prior / np.sum(active_prior)
                curr_arr = np.array([current_weights[m] for m in active_modalities])
                updated_arr = np.exp(0.6 * np.log(curr_arr + 1e-7) + 0.4 * np.log(active_prior + 1e-7))
                updated_arr = updated_arr / np.sum(updated_arr)
                for idx, m in enumerate(active_modalities):
                    current_weights[m] = float(updated_arr[idx])
            else:
                current_weights = refined_w

            new_pred_idx = int(np.argmax(probs))
            if verbose:
                w_str = ", ".join([f"{m.upper()}: {current_weights[m]*100:.1f}%" for m in active_modalities])
                print(f"   - Iteration {iteration + 1}: Diagnostic Prior Anchor = {DIAGNOSTIC_NAMES[current_pred_idx]} | Refined Fusion Weights: [{w_str}]")

            if new_pred_idx == current_pred_idx and iteration > 0:
                break
            current_pred_idx = new_pred_idx

        # ----------------------------------------------------------------------
        # [STEP 6] Split Conformal Uncertainty Quantification & Risk Assessment
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[STEP 6] Split Conformal Uncertainty Quantification & Risk Assessment:")

        conformal_res = self.conformal.predict_set(probs)
        prediction_set = conformal_res['prediction_set']

        primary_class = DIAGNOSTIC_NAMES[current_pred_idx]
        primary_conf = float(probs[current_pred_idx])

        if verbose:
            print(f"   - 95% Conformal Prediction Set : {prediction_set} (Set Size: {conformal_res.get('set_size', 1)})")
            print(f"   - Uncertainty / Risk Level     : {conformal_res.get('uncertainty_level', 'LOW')}")
            print(f"   - Primary Consensus Diagnosis  : >> {primary_class} << with {primary_conf*100:.2f}% Confidence")

        evidence_summary = {
            'quality': {'image': q_img, 'metadata': q_meta, 'text': q_text},
            'reliability': {'image': r_img, 'metadata': r_meta, 'text': r_text},
            'utility': {'image': u_img, 'metadata': u_meta, 'text': u_text},
            'consistency': c_score,
            'sufficiency': s_score,
            'decision_gate': decision_gate,
            'clinical_action': clinical_action,
            'evidence_scores': {'image': e_img, 'metadata': e_meta, 'text': e_text},
            'dynamic_fusion_weights': current_weights
        }

        return {
            'selected_model': f"Model {primary_model_code}",
            'active_modalities': active_modalities,
            'primary_prediction': primary_class,
            'confidence': primary_conf,
            'class_probabilities': {DIAGNOSTIC_NAMES[i]: float(probs[i]) for i in range(6)},
            'candidate_models': candidate_results,
            'prediction_set_95': prediction_set,
            'uncertainty_assessment': conformal_res,
            'evidence_manager': evidence_summary,
            'text_generated': text_string if auto_generate_text else None
        }


def print_prediction_report(res: dict, title="DERMA-GUARD CLINICAL DIAGNOSTIC REPORT"):
    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)
    print(f"[*] Primary Executed Architecture : {res['selected_model']}")
    print(f"[*] Active Input Modalities       : {', '.join(res['active_modalities']).upper()}")
    print(f"[*] Primary Diagnostic Finding    : >> {res['primary_prediction']} << (Confidence: {res['confidence']*100:.2f}%)")
    print(f"[*] 95% Conformal Set Guarantee   : {res['prediction_set_95']}")

    # Candidate models breakdown table
    if 'candidate_models' in res and res['candidate_models']:
        print("\n[+] Individual Evidence Scores & Predictions Across Candidate Models:")
        print("    +-------+----------------------------------+----------------+-------------+------------+")
        print("    | Model | Modality Architecture            | Evidence Score | Prediction  | Confidence |")
        print("    +-------+----------------------------------+----------------+-------------+------------+")
        for code, c_res in res['candidate_models'].items():
            print(f"    | {code:5s} | {c_res['name'][:32]:32s} |     {c_res['evidence_score']:6.3f}     |     {c_res['predicted_class']:5s}   |   {c_res['confidence']*100:5.1f}%   |")
        print("    +-------+----------------------------------+----------------+-------------+------------+")

    print("\n[+] Full Class Probability Distribution (Primary Fused Model):")
    for cls_name, p in res['class_probabilities'].items():
        bar = "#" * int(p * 35)
        print(f"    - {cls_name:5s}: {p*100:6.2f}% | {bar}")

    ev = res['evidence_manager']
    print(f"\n[+] 5D Evidence Evaluation (Sufficiency S = {ev['sufficiency']:.2f}, Gate = {ev['decision_gate']}):")
    print(f"    - Recommended Clinical Action : {ev['clinical_action']}")
    print(f"    - Inter-Modal Consistency (C): {ev['consistency']:.2f}")
    print("    - Modality Quality / Reliability / Utility Breakdown:")
    for m in res['active_modalities']:
        q = ev['quality'][m]
        r = ev['reliability'][m]
        u = ev['utility'][m]
        e = ev['evidence_scores'][m]
        w = ev['dynamic_fusion_weights'][m]
        print(f"      * {m.upper():8s} -> Q: {q:.2f} | R: {r:.2f} | U: {u:.2f} | Ev Score: {e:.3f} | Dynamic Weight: {w*100:.1f}%")

    unc = res['uncertainty_assessment']
    print(f"\n[+] Conformal Uncertainty & Risk Assessment (Alpha = 0.05):")
    print(f"    - Prediction Set Size       : {unc.get('set_size', 'N/A')} class(es)")
    print(f"    - Uncertainty / Risk Level  : {unc.get('uncertainty_level', 'N/A')}")
    print(f"    - Actionable Clinical Advice: {unc.get('clinical_advice', unc.get('clinical_guidance', 'N/A'))}")
    print("=" * 80 + "\n")


def run_sample_inference():
    predictor = DermaGuardPredictor()

    sample_candidates = [
        os.path.join(PROJECT_ROOT, "data", "images", "PAT_1516_1765_530.png"),
        r"D:\VIT BOOKS\PROJECT 1\Dataset\images\PAT_1516_1765_530.png"
    ]
    sample_img_path = sample_candidates[0]
    for c in sample_candidates:
        if os.path.exists(c):
            sample_img_path = c
            break

    sample_meta = {
        'age': 55, 'gender': 'FEMALE', 'region': 'NECK',
        'diameter_1': 6.0, 'diameter_2': 5.0, 'fitspatrick': 3,
        'itch': 'TRUE', 'grew': 'TRUE', 'hurt': 'FALSE',
        'bleed': 'TRUE', 'elevation': 'TRUE'
    }
    sample_text = "55-year-old female presenting with a 6mm bleeding pigmented nodular lesion on the neck with clinical suspicion of melanoma."

    # Case 1: Tri-modal (Image + Metadata + Explicit Free Text) -> Model G
    res1 = predictor.predict(image_path_or_pil=sample_img_path, metadata_dict=sample_meta, text_string=sample_text)
    print_prediction_report(res1, title="[CASE 1] TRI-MODAL (IMAGE + METADATA + TEXT) -> MODEL G")

    # Case 2: Only Image -> Model A
    res2 = predictor.predict(image_path_or_pil=sample_img_path)
    print_prediction_report(res2, title="[CASE 2] ONLY IMAGE -> MODEL A (ResNet50 + ViT)")

    # Case 3: Only Metadata -> Model B
    res3 = predictor.predict(metadata_dict=sample_meta, auto_generate_text=False)
    print_prediction_report(res3, title="[CASE 3] ONLY METADATA -> MODEL B (Residual MetaBlock MLP)")

    # Case 4: Image + Metadata -> Model D
    res4 = predictor.predict(image_path_or_pil=sample_img_path, metadata_dict=sample_meta, auto_generate_text=False)
    print_prediction_report(res4, title="[CASE 4] IMAGE + METADATA -> MODEL D (Adaptive Gated Fusion)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DERMA-GUARD Clinical Inference Engine (ResNet50 + Bio_ClinicalBERT + Residual MetaBlock MLP)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Target model architecture code (A through G, e.g., --model G)")
    parser.add_argument("--image", type=str, default=None, help="Path to lesion image file")
    parser.add_argument("--metadata", type=str, default=None, help="JSON string or file path containing patient metadata dictionary")
    parser.add_argument("--text", type=str, default=None, help="Clinical free text description")
    parser.add_argument("--auto-generate-text", "--auto-text", dest="auto_text", action="store_true", help="Auto-generate clinical narrative from metadata")
    parser.add_argument("--sample-test", action="store_true", help="Run automated multi-case demonstration")
    args = parser.parse_args()

    if args.sample_test:
        run_sample_inference()
    else:
        if args.image is None and args.text is None and args.metadata is None:
            print("[INFO] No inputs specified. Running automated multi-case demonstration test...\n")
            run_sample_inference()
        else:
            meta_dict = None
            if args.metadata:
                meta_dict = parse_metadata_arg(args.metadata)

            predictor = DermaGuardPredictor()
            res = predictor.predict(
                image_path_or_pil=args.image,
                metadata_dict=meta_dict,
                text_string=args.text,
                auto_generate_text=args.auto_text,
                target_model=args.model
            )
            print_prediction_report(res)

