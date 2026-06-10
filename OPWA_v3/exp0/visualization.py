import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

from config import BIN_EDGES, BIN_NAMES, NUM_BINS, TARGET_COVERAGE, OUTPUT_DIR


def plot_coverage_gap(results):
    """Bar chart: coverage rate per bin with 0.9 target line.

    exp0_coverage_gap_plot.png
    """
    bin_names_short = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]
    coverage_rates = [results["bins"][name]["coverage_rate"] for name in BIN_NAMES]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(NUM_BINS)
    bars = ax.bar(x, coverage_rates, width=0.6, color="steelblue", edgecolor="navy")
    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target $1-\\alpha$ = {TARGET_COVERAGE:.0%}")

    # Annotate bar tops
    for bar, rate in zip(bars, coverage_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_names_short, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Transmittance Bin")
    ax.set_title("Coverage Rate by Transmittance Bin\n(Standard Split CP, $\\alpha=0.1$)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    filepath = os.path.join(OUTPUT_DIR, "exp0_coverage_gap_plot.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_transmittance_vis(per_image_results, num_samples=5):
    """First N test images: foggy image + transmittance heatmap side-by-side.

    exp0_transmittance_vis.png
    """
    num_show = min(num_samples, len(per_image_results))
    fig, axes = plt.subplots(num_show, 2, figsize=(14, 4 * num_show))

    if num_show == 1:
        axes = axes.reshape(1, -1)

    for row in range(num_show):
        result = per_image_results[row]
        img_path = result["path"]
        t_map = result["t_map"]

        # Original foggy image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Foggy Image {row + 1}\nmean_t={result['mean_t']:.3f}")
        axes[row, 0].axis("off")

        # Transmittance heatmap
        im = axes[row, 1].imshow(t_map, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"Transmittance t(x)\nbin0_ratio={result['bin0_ratio']:.2%}"
        )

        # Overlay bin contour lines at 0.2, 0.4, 0.6, 0.8
        for level in [0.2, 0.4, 0.6, 0.8]:
            axes[row, 1].contour(
                t_map, levels=[level],
                colors="white", linewidths=0.5, alpha=0.6
            )
        axes[row, 1].axis("off")

    # Color bar for the last row
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Transmittance t(x)")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.suptitle("Transmittance Visualization: Foggy Image vs Estimated t(x)",
                 fontsize=14, y=1.01)

    filepath = os.path.join(OUTPUT_DIR, "exp0_transmittance_vis.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_score_distribution(results, bin_arrays):
    """Violin/box plot of nonconformity scores per bin.

    exp0_score_distribution.png

    Note: bin_arrays doesn't store individual scores, so we regenerate
    rough distributions from mean scores. For a proper detailed plot
    we need per-bin score arrays — but we can show the mean scores
    as a bar chart with overlay.
    """
    # Since we didn't store per-pixel scores per bin (too much memory),
    # we plot: mean_score per bin as a bar, annotated with coverage_rate
    bin_names_short = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]
    mean_scores = [results["bins"][name]["mean_score"] for name in BIN_NAMES]
    coverage_rates = [results["bins"][name]["coverage_rate"] for name in BIN_NAMES]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(NUM_BINS)
    bars = ax1.bar(x, mean_scores, width=0.5, color="salmon", edgecolor="darkred", alpha=0.8,
                   label="Mean Score")

    # Annotate
    for bar, ms in zip(bars, mean_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{ms:.4f}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_names_short, rotation=30, ha="right")
    ax1.set_ylabel("Mean Nonconformity Score")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Transmittance Bin")
    ax1.set_title("Nonconformity Score by Transmittance Bin")
    ax1.grid(axis="y", alpha=0.3)

    # Overlay coverage rate as a line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, coverage_rates, "bo-", linewidth=2, markersize=8, label="Coverage Rate")
    ax2.axhline(y=TARGET_COVERAGE, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Coverage Rate")
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    filepath = os.path.join(OUTPUT_DIR, "exp0_score_distribution.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def generate_all_plots(results, per_image_results, bin_arrays):
    """Generate all 3 output plots."""
    p1 = plot_coverage_gap(results)
    p2 = plot_transmittance_vis(per_image_results, num_samples=5)
    p3 = plot_score_distribution(results, bin_arrays)
    return [p1, p2, p3]
