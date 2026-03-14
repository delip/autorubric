"""Generate the few-shot calibration comparison chart for the cookbook."""

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

METRICS = ["Accuracy", "Cohen's Kappa"]
BASELINE = [75.0, 41.2]
CALIBRATED = [90.0, 78.2]

# Original-scale kappa values for annotation
KAPPA_BASELINE = 0.412
KAPPA_CALIBRATED = 0.782

COLOR_BASELINE = "#6c8ebf"
COLOR_CALIBRATED = "#82b366"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(METRICS))
    width = 0.3

    with THEME:
        fig, ax = plt.subplots(figsize=(5, 4))

        bars_base = ax.bar(x - width / 2, BASELINE, width, label="Baseline",
                           color=COLOR_BASELINE, edgecolor="0.95")
        bars_cal = ax.bar(x + width / 2, CALIBRATED, width, label="Calibrated (3-shot)",
                          color=COLOR_CALIBRATED, edgecolor="0.95")

        # Accuracy annotations (with %)
        ax.text(bars_base[0].get_x() + bars_base[0].get_width() / 2,
                bars_base[0].get_height() + 1.5, "75.0%",
                ha="center", va="bottom", fontsize=6)
        ax.text(bars_cal[0].get_x() + bars_cal[0].get_width() / 2,
                bars_cal[0].get_height() + 1.5, "90.0%",
                ha="center", va="bottom", fontsize=6)

        # Kappa annotations (original scale)
        ax.text(bars_base[1].get_x() + bars_base[1].get_width() / 2,
                bars_base[1].get_height() + 1.5, f"{KAPPA_BASELINE}",
                ha="center", va="bottom", fontsize=6)
        ax.text(bars_cal[1].get_x() + bars_cal[1].get_width() / 2,
                bars_cal[1].get_height() + 1.5, f"{KAPPA_CALIBRATED}",
                ha="center", va="bottom", fontsize=6)

        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS)
        ax.set_ylim(0, 105)

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False)
        fig.suptitle("Few-Shot Calibration: Baseline vs Calibrated",
                     fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.85)

        out = IMAGE_DIR / "fewshot-calibration-comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
