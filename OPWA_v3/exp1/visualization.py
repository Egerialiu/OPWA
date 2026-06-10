"""Experiment 1-3 Visualization.

Generates:
  1. exp1_coverage_gap_plot.png — PS-CP per-bin coverage bars
  2. exp1_comparison_plot.png — Standard CP vs PS-CP side-by-side
  3. exp1_calib_histogram.png — calibration set t distribution
  4. exp2_baseline_comparison.png — all methods comparison
  5. exp3_ablation_beta.png — β sweep results
  6. exp3_ablation_nbins.png — Bin count sweep results
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
from config import BIN_EDGES, BIN_NAMES, NUM_BINS, TARGET_COVERAGE

EXP1_OUTPUT_DIR = "/root/opwa_v3/exp1_outputs/"
os.makedirs(EXP1_OUTPUT_DIR, exist_ok=True)

# Color scheme
STD_COLOR = "#4A7BB5"  # steel blue
PSCP_COLOR = "#D4504A"  # coral/red
WCP_COLOR = "#7AC36A"  # green
TS_COLOR = "#E0A34A"  # orange


def load_results(path):
    """Load a JSON results file."""
    with open(path) as f:
        return json.load(f)


def plot_pscp_coverage(pscp_results):
    """Plot PS-CP per-bin coverage bars.

    Args:
        pscp_results: dict from pscp_evaluation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bin_labels = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]
    cov_rates = [pscp_results["bins"][n]["coverage_rate"] for n in BIN_NAMES]
    gaps = [pscp_results["bins"][n]["gap"] for n in BIN_NAMES]

    x = np.arange(NUM_BINS)
    bars = ax.bar(x, cov_rates, width=0.6, color=PSCP_COLOR, edgecolor="darkred")

    # Annotate bars with coverage rate and gap
    for bar, rate, gap in zip(bars, cov_rates, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.3f}\n(gap={gap:.3f})", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target 1-α = {TARGET_COVERAGE:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Transmittance Bin")
    ax.set_title("PS-CP: Coverage Rate by Transmittance Bin\n(Stratified q_hat per bin)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_coverage_gap_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_comparison(standard_results, pscp_results):
    """Side-by-side bar chart: Standard CP vs PS-CP.

    Args:
        standard_results: dict from exp0.
        pscp_results: dict from exp1.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bin_labels = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]

    std_rates = [standard_results["bins"][n]["coverage_rate"] for n in BIN_NAMES]
    pscp_rates = [pscp_results["bins"][n]["coverage_rate"] for n in BIN_NAMES]

    x = np.arange(NUM_BINS)
    width = 0.35

    bars1 = ax.bar(x - width / 2, std_rates, width, label="Standard CP",
                   color=STD_COLOR, edgecolor="navy")
    bars2 = ax.bar(x + width / 2, pscp_rates, width, label="PS-CP (Ours)",
                   color=PSCP_COLOR, edgecolor="darkred")

    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target {TARGET_COVERAGE:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Transmittance Bin")
    ax.set_title("Coverage Rate: Standard CP vs PS-CP\n(Foggy Cityscapes, α=0.1)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_comparison_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_calib_histogram(per_image_t_data):
    """Plot calibration set transmittance histogram.

    Args:
        per_image_t_data: list of dicts with t_mean per calibration image,
                          or path to the JSON file.
    """
    if isinstance(per_image_t_data, str):
        with open(per_image_t_data) as f:
            per_image_t_data = json.load(f)

    t_means = np.array([d["t_mean"] for d in per_image_t_data])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of per-image mean t
    axes[0].hist(t_means, bins=30, color="steelblue", edgecolor="navy", alpha=0.7)
    axes[0].set_xlabel("Mean Transmittance per Image")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_title("Calibration Set: Per-Image Mean Transmittance\n(Cityscapes val, 250 images)")
    axes[0].axvline(t_means.mean(), color="red", linestyle="--",
                    label=f"Mean: {t_means.mean():.4f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(t_means, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5))
    axes[1].set_ylabel("Mean Transmittance")
    axes[1].set_title(f"Distribution Summary\nMedian: {np.median(t_means):.4f}, "
                      f"Std: {t_means.std():.4f}")
    axes[1].grid(alpha=0.3)

    # Add bin edges
    for edge in BIN_EDGES:
        if edge <= 1.0:
            axes[1].axhline(y=edge, color="orange", linestyle=":", alpha=0.5,
                            label=f"Bin edge: {edge:.1f}" if edge in (0.2, 0.4, 0.6, 0.8) else "")

    fig.tight_layout()
    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_calib_histogram.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_baseline_comparison(results_dict):
    """Plot all methods side by side.

    Args:
        results_dict: dict mapping method_name -> results dict.
    """
    methods = list(results_dict.keys())
    colors = [STD_COLOR, PSCP_COLOR, WCP_COLOR, TS_COLOR][:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: per-bin coverage rates
    bin_labels = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]
    x = np.arange(NUM_BINS)
    width = 0.8 / len(methods)

    for i, (method, res) in enumerate(results_dict.items()):
        rates = [res["bins"][n]["coverage_rate"] for n in BIN_NAMES]
        offset = (i - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, rates, width, label=method, color=colors[i], alpha=0.85)

    axes[0].axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bin_labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Coverage Rate")
    axes[0].set_xlabel("Transmittance Bin")
    axes[0].set_title("Per-Bin Coverage by Method")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: metrics table as bar chart
    metrics = ["Overall\nCoverage", "Max Gap", "Bin0 Gap", "Mean |C|"]
    metric_values = {m: [] for m in metrics}
    for method, res in results_dict.items():
        bins_data = res["bins"]
        mean_set_size = np.mean([bins_data[n]["mean_set_size"] for n in BIN_NAMES])
        metric_values["Overall\nCoverage"].append(res.get("overall_test_coverage", 0))
        metric_values["Max Gap"].append(res.get("max_gap", 0))
        metric_values["Bin0 Gap"].append(res.get("bin0_gap", 0))
        metric_values["Mean |C|"].append(mean_set_size)

    x2 = np.arange(len(metrics))
    width2 = 0.8 / len(methods)
    for i, method in enumerate(methods):
        vals = [metric_values[m][i] for m in metrics]
        offset = (i - (len(methods) - 1) / 2) * width2
        axes[1].bar(x2 + offset, vals, width2, label=method, color=colors[i], alpha=0.85)

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics, rotation=15)
    axes[1].set_ylabel("Value")
    axes[1].set_title("Summary Metrics by Method")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp2_baseline_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_ablation_beta(beta_results):
    """Plot β sweep: bin0_gap vs β.

    Args:
        beta_results: list of dicts, each with "beta", "bin0_gap", "max_gap".
    """
    betas = [r["beta"] for r in beta_results]
    bin0_gaps = [r["bin0_gap"] for r in beta_results]
    max_gaps = [r["max_gap"] for r in beta_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(betas, bin0_gaps, "o-", color=PSCP_COLOR, linewidth=2, label="Bin0 Gap")
    ax.plot(betas, max_gaps, "s--", color=STD_COLOR, linewidth=2, label="Max Gap")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("β (Beer-Lambert attenuation coefficient)")
    ax.set_ylabel("Coverage Gap")
    ax.set_title("Ablation: Effect of β on PS-CP Coverage Gaps")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp3_ablation_beta.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_ablation_nbins(nbins_results):
    """Plot bin count sweep: max_gap vs N bins.

    Args:
        nbins_results: list of dicts, each with "n_bins", "bin0_gap", "max_gap".
    """
    nbins = [r["n_bins"] for r in nbins_results]
    bin0_gaps = [r["bin0_gap"] for r in nbins_results]
    max_gaps = [r["max_gap"] for r in nbins_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nbins, bin0_gaps, "o-", color=PSCP_COLOR, linewidth=2, label="Bin0 Gap")
    ax.plot(nbins, max_gaps, "s--", color=STD_COLOR, linewidth=2, label="Max Gap")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Stratification Bins")
    ax.set_ylabel("Coverage Gap")
    ax.set_title("Ablation: Effect of Bin Count on PS-CP Coverage Gaps")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(nbins)

    fig.tight_layout()
    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp3_ablation_nbins.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def generate_all_plots():
    """Generate all available plots from saved results."""
    plots = []

    # Experiment 1 plots
    exp1_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_results.json")
    exp0_path = "/root/opwa_v3/exp0_outputs/exp0_results.json"

    if os.path.exists(exp1_path):
        pscp_results = load_results(exp1_path)
        plots.append(plot_pscp_coverage(pscp_results))

        if os.path.exists(exp0_path):
            standard_results = load_results(exp0_path)
            plots.append(plot_comparison(standard_results, pscp_results))

    # Calibration histogram
    calib_t_path = os.path.join(EXP1_OUTPUT_DIR, "calib_per_image_t.json")
    if os.path.exists(calib_t_path):
        plots.append(plot_calib_histogram(calib_t_path))

    # Baseline comparison
    baseline_results = {}
    for method_name, filename in [
        ("Standard CP", exp0_path),
        ("PS-CP", exp1_path),
        ("Weighted CP", os.path.join(EXP1_OUTPUT_DIR, "exp2_wcp_results.json")),
        ("Temp Scaling CP", os.path.join(EXP1_OUTPUT_DIR, "exp2_temperature_cp_results.json")),
    ]:
        if os.path.exists(filename):
            baseline_results[method_name] = load_results(filename)

    if len(baseline_results) >= 2:
        plots.append(plot_baseline_comparison(baseline_results))

    return plots


if __name__ == "__main__":
    generate_all_plots()
