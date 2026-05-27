"""Generate charts for the held-out rubric improvement cookbook recipe."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme

IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"

THEME = (
    load_theme("scientific")
    .set_ticks(size_major=4, size_minor=2, width_major=0.8, width_minor=0.5)
    .set_grid(draw=False)
    .set_font(family="sans-serif", size=8)
    .set_axis_labels(pad=8)
)

CRITERIA = [
    "paper_summary",
    "methodology_assessment",
    "statistical_evaluation",
    "strengths_and_weaknesses",
    "constructive_tone",
    "structured_format",
    "specific_references",
    "concise_review",
    "clear_recommendation",
    "factual_misrepresentation",
]

SEED_ACCURACY = [70, 50, 60, 40, 50, 60, 40, 80, 50, 90]
IMPROVED_ACCURACY = [100, 90, 90, 90, 80, 90, 80, 90, 100, 100]

SEED_FP = [30, 40, 30, 50, 30, 30, 40, 10, 40, 0]
SEED_FN = [0, 10, 10, 10, 20, 10, 20, 10, 10, 10]

CONVERGENCE_ITERS = [0, 1, 2, 3, 4]
CONVERGENCE_ACC = [0.59, 0.75, 0.84, 0.91, 0.93]

COLOR_CORAL = "#e76f51"
COLOR_TEAL = "#2a9d8f"
COLOR_DARK = "#264653"


def accuracy_before_after() -> None:
    """Grouped bar chart: per-criterion accuracy before vs after."""
    x = np.arange(len(CRITERIA))
    width = 0.35

    with THEME:
        fig, ax = plt.subplots(figsize=(8, 4))

        bars_seed = ax.bar(
            x - width / 2,
            SEED_ACCURACY,
            width,
            label="Seed",
            color=COLOR_CORAL,
            edgecolor="0.95",
        )
        bars_improved = ax.bar(
            x + width / 2,
            IMPROVED_ACCURACY,
            width,
            label="Improved",
            color=COLOR_TEAL,
            edgecolor="0.95",
        )

        for bar, val in zip(bars_seed, SEED_ACCURACY):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=5,
            )
        for bar, val in zip(bars_improved, IMPROVED_ACCURACY):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=5,
            )

        ax.set_ylabel("% correct")
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(CRITERIA, rotation=25, ha="right")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )

        out = IMAGE_DIR / "held-out-accuracy-before-after.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


def error_rates() -> None:
    """Horizontal stacked bar chart: FP/FN rates for seed rubric."""
    total_error = [fp + fn for fp, fn in zip(SEED_FP, SEED_FN)]
    order = sorted(range(len(CRITERIA)), key=lambda i: total_error[i])

    names = [CRITERIA[i] for i in order]
    fp_sorted = [SEED_FP[i] for i in order]
    fn_sorted = [SEED_FN[i] for i in order]

    y = np.arange(len(names))

    with THEME:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        bars_fp = ax.barh(
            y,
            fp_sorted,
            height=0.6,
            label="False positive rate",
            color=COLOR_CORAL,
            edgecolor="0.95",
        )
        bars_fn = ax.barh(
            y,
            fn_sorted,
            height=0.6,
            left=fp_sorted,
            label="False negative rate",
            color=COLOR_DARK,
            edgecolor="0.95",
        )

        for bar, val in zip(bars_fp, fp_sorted):
            if val >= 15:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
        for bar, val, left in zip(bars_fn, fn_sorted, fp_sorted):
            if val >= 10:
                ax.text(
                    left + val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )

        ax.set_xlabel("% of graded items")
        ax.set_xlim(0, 65)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )

        out = IMAGE_DIR / "held-out-error-rates.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


def accuracy_convergence() -> None:
    """Line chart: mean accuracy across iterations."""
    with THEME:
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(
            CONVERGENCE_ITERS,
            CONVERGENCE_ACC,
            marker="o",
            color=COLOR_TEAL,
            linewidth=1.8,
            markersize=5,
        )

        for x, y in zip(CONVERGENCE_ITERS, CONVERGENCE_ACC):
            ax.text(
                x,
                y + 0.025,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        ax.axhline(0.90, linestyle="--", color="0.5", linewidth=0.8)
        ax.text(
            CONVERGENCE_ITERS[-1] + 0.15,
            0.90,
            "Target",
            va="center",
            fontsize=6,
            color="0.5",
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean accuracy")
        ax.set_xlim(-0.3, 4.5)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(CONVERGENCE_ITERS)

        out = IMAGE_DIR / "held-out-accuracy-convergence.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    accuracy_before_after()
    error_rates()
    accuracy_convergence()


if __name__ == "__main__":
    main()
