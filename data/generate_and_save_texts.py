import os
import sys
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.text_generator import generate_clinical_text, compute_ptis

def clean_synthetic_description(desc: str, row: pd.Series) -> str:
    """
    Cleans synthetic text descriptions by replacing unmeasured NaN tokens with
    structured dimensions from metadata or natural medical descriptors.
    """
    if not isinstance(desc, str) or not desc.strip():
        return ""
    
    d1 = row.get('diameter_1', None)
    d2 = row.get('diameter_2', None)
    if pd.notna(d1) and pd.notna(d2) and float(d1) > 0 and float(d2) > 0:
        dim_str = f"{float(d1):.1f} by {float(d2):.1f} millimeters"
    elif pd.notna(d1) and float(d1) > 0:
        dim_str = f"{float(d1):.1f} millimeters across"
    else:
        dim_str = "unmeasured dimensions"

    t = desc
    t = re.sub(r'about nan by nan millimeters\.?', f'dimensions: {dim_str}.', t, flags=re.IGNORECASE)
    t = re.sub(r'roughly nanmm across\.?', f'dimensions: {dim_str}.', t, flags=re.IGNORECASE)
    t = re.sub(r'measures around nan mm\.?', f'dimensions: {dim_str}.', t, flags=re.IGNORECASE)
    t = re.sub(r"it's about nan by nan millimeters\.?", f'lesion dimensions: {dim_str}.', t, flags=re.IGNORECASE)
    t = re.sub(r'\bnan\b', 'unspecified', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def generate_and_export_clinical_texts(metadata_path=None, output_dir=None):
    """
    Generates rich, authentic clinical free-text descriptions for all patient records
    in the PAD-UFES-20 dataset using the provided synthetic text dataset (synthetic_texts.csv)
    and clinical metadata, exporting to CSV files.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(output_dir, exist_ok=True)

    if metadata_path is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "metadata.csv"),
            r"D:\VIT BOOKS\PROJECT 1\Dataset\metadata.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                metadata_path = c
                break

    if metadata_path is None or not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata file not found in any candidate path.")

    print(f"[*] Reading dataset metadata from: {metadata_path}", flush=True)
    df = pd.read_csv(metadata_path)
    print(f"[*] Total patient records to process: {len(df)}", flush=True)

    # Check for user-provided synthetic texts dataset
    synthetic_csv = os.path.join(output_dir, "synthetic_texts.csv")
    has_synthetic = os.path.exists(synthetic_csv)
    syn_map = {}
    if has_synthetic:
        print(f"[*] Found user-provided synthetic texts dataset at: {synthetic_csv}", flush=True)
        df_syn = pd.read_csv(synthetic_csv)
        if 'img_id' in df_syn.columns:
            syn_map = df_syn.set_index('img_id').to_dict(orient='index')
            print(f"    Loaded {len(syn_map)} synthetic text records.", flush=True)

    print("[*] Synthesizing comprehensive clinical narratives from synthetic data & metadata...", flush=True)
    clinical_texts = []
    ptis_scores = []
    word_counts = []

    for idx, (_, row) in enumerate(df.iterrows()):
        img_id = str(row.get('img_id', ''))
        syn_entry = syn_map.get(img_id, None)

        if syn_entry is not None:
            paraphrased = clean_synthetic_description(str(syn_entry.get('paraphrased_description', '')), row)
            filled = clean_synthetic_description(str(syn_entry.get('filled_description', '')), row)
            if paraphrased and filled:
                text = f"Patient Narrative: {paraphrased} Clinical Details: {filled}"
            elif paraphrased:
                text = f"Patient Narrative: {paraphrased}"
            elif filled:
                text = f"Clinical Presentation: {filled}"
            else:
                text = generate_clinical_text(row, include_morphology=True)
        else:
            text = generate_clinical_text(row, include_morphology=True)

        ptis = compute_ptis(text)
        wc = len(text.split())
        clinical_texts.append(text)
        ptis_scores.append(ptis)
        word_counts.append(wc)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"    Processed {idx + 1}/{len(df)} clinical narratives (Avg word count: {np.mean(word_counts):.1f} words)...", flush=True)

    # 1. Dedicated CSV with generated texts
    df_texts = pd.DataFrame({
        'img_id': df['img_id'] if 'img_id' in df.columns else np.arange(len(df)),
        'patient_id': df['patient_id'] if 'patient_id' in df.columns else df.index,
        'diagnostic': df['diagnostic'] if 'diagnostic' in df.columns else 'UNKNOWN',
        'clinical_text': clinical_texts,
        'word_count': word_counts,
        'ptis_quality_score': ptis_scores
    })

    texts_csv_path = os.path.join(output_dir, "generated_clinical_texts.csv")
    df_texts.to_csv(texts_csv_path, index=False)
    print(f"[SAVED] Generated synthetic clinical texts saved to: {texts_csv_path}", flush=True)

    # 2. Merged metadata CSV with clinical texts
    df_merged = df.copy()
    df_merged['clinical_text'] = clinical_texts
    df_merged['word_count'] = word_counts
    df_merged['ptis_quality_score'] = ptis_scores

    merged_csv_path = os.path.join(output_dir, "metadata_with_clinical_text.csv")
    df_merged.to_csv(merged_csv_path, index=False)
    print(f"[SAVED] Full metadata with synthetic clinical texts saved to: {merged_csv_path}", flush=True)

    return texts_csv_path, merged_csv_path


if __name__ == '__main__':
    generate_and_export_clinical_texts()
