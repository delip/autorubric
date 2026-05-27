"""Generate a horizontal bar chart comparing CANNOT_ASSESS strategy scores."""

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

STRATEGIES = ["SKIP", "PARTIAL (0.5)", "ZERO", "FAIL"]
SCORES = [0.85, 0.78, 0.72, 0.60]
COLORS = ["#2a9d8f", "#61a5c2", "#f4a261", "#e76f51"]


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by score descending so highest bar appears at top
    ordered = sorted(zip(STRATEGIES, SCORES, COLORS), key=lambda t: t[1])
    names, vals, colors = zip(*ordered)

    with THEME:
        fig, ax = plt.subplots(figsize=(5, 3))

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="0.95", height=0.55)

        for bar, val in zip(bars, vals):
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=6,
                fontweight="bold",
            )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Score")
        ax.set_title("CANNOT_ASSESS Strategy Score Comparison", fontweight="bold")

        out = IMAGE_DIR / "cannot-assess-strategy-comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
