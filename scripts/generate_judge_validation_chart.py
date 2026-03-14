"""Generate a per-criterion validation metrics chart for a content moderation judge."""

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

CRITERIA = ["hate_speech", "harassment", "misinformation", "self_harm", "spam"]
METRICS = {
    "Accuracy": [92.0, 87.0, 78.0, 95.0, 89.0],
    "Kappa":    [81.2, 71.4, 52.1, 87.6, 76.1],
    "F1":       [89.1, 82.3, 74.2, 92.3, 85.6],
}
METRIC_COLORS = {"Accuracy": "#4682B4", "Kappa": "#E8825A", "F1": "#2A9D8F"}


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    labels = [name.replace("_", " ").title() for name in CRITERIA]
    x = np.arange(len(CRITERIA))
    width = 0.22
    metric_names = list(METRICS.keys())

    with THEME:
        fig, ax = plt.subplots(figsize=(7, 4))

        # Highlight the misinformation group with a subtle background
        misinfo_idx = CRITERIA.index("misinformation")
        ax.axvspan(
            misinfo_idx - 0.42, misinfo_idx + 0.42,
            color="#fff3cd", zorder=0,
        )

        for i, name in enumerate(metric_names):
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset, METRICS[name], width,
                label=name, color=METRIC_COLORS[name], edgecolor="0.95",
            )
            for bar, val in zip(bars, METRICS[name]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=5,
                )

        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 108)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=3, frameon=False,
        )
        fig.suptitle("Per-Criterion Validation Metrics", fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.85)

        out = IMAGE_DIR / "judge-validation-per-criterion.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
