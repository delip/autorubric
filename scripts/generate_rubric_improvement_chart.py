"""Generate the rubric improvement dual-axis chart for the cookbook."""

from pathlib import Path

import matplotlib.pyplot as plt
from aquarel import load_theme

IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"

THEME = (
    load_theme("scientific")
    .set_ticks(size_major=4, size_minor=2, width_major=0.8, width_minor=0.5)
    .set_grid(draw=False)
    .set_font(family="sans-serif", size=8)
    .set_axis_labels(pad=8)
)

# Data from the cookbook article table
ITERATIONS = [0, 1, 2, 3, 4]
QUALITY_SCORES = [0.00, 0.89, 0.92, 0.95, 1.00]
ISSUES = [21, 2, 1, 1, 0]

COLOR_BARS = "#e76f51"
COLOR_LINE = "#2a9d8f"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with THEME:
        fig, ax1 = plt.subplots(figsize=(7, 4))

        # Bars: issues detected (left y-axis)
        bars = ax1.bar(ITERATIONS, ISSUES, width=0.5, color=COLOR_BARS,
                       edgecolor="0.95", label="Issues detected", zorder=2)
        for bar, val in zip(bars, ISSUES):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontsize=6)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Issues detected")
        ax1.set_ylim(0, 26)
        ax1.set_xticks(ITERATIONS)

        # Line: quality score (right y-axis)
        ax2 = ax1.twinx()
        # Aquarel scientific theme hides the right spine; re-enable for dual-axis
        ax2.spines["right"].set_visible(True)
        ax2.plot(ITERATIONS, QUALITY_SCORES, marker="o", color=COLOR_LINE,
                 linewidth=1.8, markersize=5, label="Quality score", zorder=3)
        for x, y in zip(ITERATIONS, QUALITY_SCORES):
            ax2.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                     fontsize=6, color=COLOR_LINE)

        ax2.set_ylabel("Quality score")
        ax2.set_ylim(-0.05, 1.20)

        # Combined legend above chart
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)

        fig.suptitle("Rubric Improvement Over Iterations", fontweight="bold",
                     y=0.98)
        fig.subplots_adjust(top=0.85)

        out = IMAGE_DIR / "rubric-improvement-chart.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
