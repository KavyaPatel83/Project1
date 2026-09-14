"""
DERMA-GUARD: Standalone Model G Multi-Modal Inference Demo
Usage:
    python demo_predict.py
"""

import os
import sys

# Ensure root directory is on Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import DermaGuardPredictor

def main():
    print("[*] Initializing DermaGuard Multi-Modal Predictor...", flush=True)
    predictor = DermaGuardPredictor()

    # Automatically resolve image path
    candidates = [
        r"D:\VIT BOOKS\PROJECT 1\Dataset\images\PAT_1516_1765_530.png",
        os.path.join(PROJECT_ROOT, "data", "images", "PAT_1516_1765_530.png"),
        os.path.join(PROJECT_ROOT, "data", "images")
    ]
    img_path = None
    for c in candidates:
        if os.path.isfile(c):
            img_path = c
            break
        elif os.path.isdir(c):
            files = [f for f in os.listdir(c) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                img_path = os.path.join(c, files[0])
                break

    if not img_path:
        print("[!] Warning: No sample image found. Please provide an image path.")
        return

    print(f"[*] Input Cutaneous Image: {img_path}")

    # Patient metadata record
    metadata = {
        'age': 55,
        'gender': 'FEMALE',
        'region': 'NECK',
        'diameter_1': 6.0,
        'diameter_2': 5.0,
        'fitspatrick': 3,
        'itch': 'TRUE',
        'grew': 'TRUE',
        'hurt': 'FALSE',
        'bleed': 'TRUE',
        'elevation': 'TRUE'
    }

    # Clinical presentation narrative
    clinical_text = "55-year-old female presenting with a 6mm bleeding pigmented nodular lesion on the neck."

    print("\n[*] Executing Full Tri-Modal Model G Inference...")
    result = predictor.predict(
        image_path_or_pil=img_path,
        metadata_dict=metadata,
        text_string=clinical_text,
        auto_generate_text=True
    )

    print("\n" + "=" * 70)
    print("                    DERMA-GUARD INFERENCE RESULT")
    print("=" * 70)
    print(f"Primary Diagnosis         : {result['primary_prediction']}")
    print(f"Confidence Score          : {result['confidence'] * 100:.2f}%")
    print(f"95% Conformal Set         : {result['prediction_set_95']}")
    print(f"Evidence Decision Gate    : {result['evidence_manager']['decision_gate']}")
    print(f"Clinical Action           : {result['evidence_manager']['clinical_action']}")
    print(f"Dynamic Fusion Weights    :")
    for mod, w in result['evidence_manager']['dynamic_fusion_weights'].items():
        print(f"  - {mod.upper():<10}: {w * 100:.1f}%")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
