"""
PHYS-DIFF PART 2: PAPER VISUALIZATIONS & METRIC SUMMARY
Generates IEEE-style comparative plots and feature importance charts.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 1. Load Generated Benchmarks
comp_path = RESULTS_DIR / "baseline_vs_rf_comparison.csv"
feat_path = RESULTS_DIR / "feature_importance.csv"

assert comp_path.exists(), f"Missing {comp_path}"
assert feat_path.exists(), f"Missing {feat_path}"

df_comp = pd.read_csv(comp_path)
df_feat = pd.read_csv(feat_path)

print("\n" + "=" * 65)
print("     PHYS-DIFF PART 2: RESEARCH BENCHMARK SUMMARY TABLE")
print("=" * 65)
print(df_comp.to_string(index=False))
print("=" * 65)

# 2. Plot Feature Importance (IEEE single-column style)
plt.figure(figsize=(7, 4.5), dpi=300)
df_feat_sorted = df_feat.sort_values(by="Importance", ascending=True)

colors = ["#2b5c8f" if "RATIO" in f or "NDSI" in f else "#5c768d" for f in df_feat_sorted["Feature"]]
bars = plt.barh(df_feat_sorted["Feature"], df_feat_sorted["Importance"], color=colors, edgecolor="black", linewidth=0.6)

plt.xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
plt.title("Sentinel-2 Multispectral Feature Importance (Snow Discrimination)", fontsize=11, fontweight="bold", pad=12)
plt.grid(axis="x", linestyle="--", alpha=0.6)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.003, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", ha="left", va="center", fontsize=8.5)

plt.tight_layout()
feat_plot_path = FIGURES_DIR / "figure_feature_importance.png"
plt.savefig(feat_plot_path)
plt.close()
print(f"[✓] Generated Figure: {feat_plot_path}")

# 3. Plot Baseline vs RF Metrics Comparison
metrics = ["Precision", "Recall", "F1", "IoU"]
baseline_vals = df_comp[df_comp["Method"].str.contains("Baseline")][metrics].values[0]
rf_vals = df_comp[df_comp["Method"].str.contains("Random Forest")][metrics].values[0]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(6.5, 4.2), dpi=300)
plt.bar([p - width/2 for p in x], baseline_vals, width=width, label="Baseline (NDSI >= 0.495)", color="#b23b3b", edgecolor="black", linewidth=0.6)
plt.bar([p + width/2 for p in x], rf_vals, width=width, label="Multispectral Random Forest", color="#2e7d32", edgecolor="black", linewidth=0.6)

plt.xticks(x, metrics, fontsize=10, fontweight="bold")
plt.ylabel("Score", fontsize=11, fontweight="bold")
plt.ylim(0, 1.15)
plt.title("Baseline NDSI vs. Proposed Multispectral Classifier", fontsize=11, fontweight="bold", pad=12)
plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Value annotations
for i in range(len(metrics)):
    plt.text(i - width/2, baseline_vals[i] + 0.02, f"{baseline_vals[i]:.3f}", ha="center", fontsize=8.5)
    plt.text(i + width/2, rf_vals[i] + 0.02, f"{rf_vals[i]:.3f}", ha="center", fontsize=8.5)

plt.tight_layout()
comp_plot_path = FIGURES_DIR / "figure_baseline_vs_rf.png"
plt.savefig(comp_plot_path)
plt.close()
print(f"[✓] Generated Figure: {comp_plot_path}")