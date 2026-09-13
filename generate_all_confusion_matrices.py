"""
DERMA-GUARD: Standalone Confusion Matrix Heatmap Generator for All Models
Reads saved model benchmark results and generates unified publication-quality heatmaps.
"""

import os
import pandas as pd
from utils.plotting import plot_all_models_confusion_matrices

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "saved_models", "models_benchmark_summary.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark summary file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    benchmark_results = df.to_dict(orient='records')

    save_dir = os.path.join(base_dir, "saved_models")
    out_png = os.path.join(save_dir, "all_models_confusion_matrices.png")
    out_jpg = os.path.join(save_dir, "all_models_confusion_matrices.jpg")

    print(f"Generating unified confusion matrix heatmaps for {len(benchmark_results)} models...")
    plot_all_models_confusion_matrices(benchmark_results, out_png)
    plot_all_models_confusion_matrices(benchmark_results, out_jpg)

    print(f"Heatmap figures successfully generated:")
    print(f" - [PNG] {out_png}")
    print(f" - [JPG] {out_jpg}")

if __name__ == '__main__':
    main()
