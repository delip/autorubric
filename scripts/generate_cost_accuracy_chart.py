"""Generate a cost vs accuracy scatter plot for different models."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from aquarel import load_theme

IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"

THEME = (
    load_theme("scientific")
    .set_ticks(size_major=4, size_minor=2, width_major=0.8, width_minor=0.5)
    .set_grid(draw=False)
    .set_font(family="sans-serif", size=8)
    .set_axis_labels(pad=8)
)

MODELS = [
    {
        "name": "Gemini Flash",
        "accuracy": 87.5,
        "cost": 0.0012,
        "color": "#e76f51",
        "offset": (8, -10),
    },
    {
        "name": "Claude Haiku",
        "accuracy": 89.2,
        "cost": 0.0018,
        "color": "#2a9d8f",
        "offset": (8, 6),
    },
    {"name": "GPT-4 Mini", "accuracy": 91.3, "cost": 0.0034, "color": "#264653", "offset": (-8, 6)},
    {
        "name": "GPT-4 Turbo",
        "accuracy": 94.1,
        "cost": 0.0156,
        "color": "#e9c46a",
        "offset": (-8, -10),
    },
]


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by cost for the Pareto frontier line
    pareto = sorted(MODELS, key=lambda m: m["cost"])

    with THEME:
        fig, ax = plt.subplots(figsize=(5.5, 4))

        for m in MODELS:
            ax.scatter(m["cost"], m["accuracy"], marker="o", s=80, color=m["color"], zorder=3)
            ax.annotate(
                m["name"],
                xy=(m["cost"], m["accuracy"]),
                xytext=m["offset"],
                textcoords="offset points",
                fontsize=6,
                ha="right" if m["offset"][0] < 0 else "left",
                va="bottom" if m["offset"][1] > 0 else "top",
            )

        # Pareto frontier (all 4 models form the frontier)
        ax.plot(
            [m["cost"] for m in pareto],
            [m["accuracy"] for m in pareto],
            linestyle="--",
            color="0.75",
            linewidth=1,
            zorder=1,
        )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:g}"))
        ax.set_xlabel("Cost ($)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Cost vs Accuracy", fontweight="bold")

        out = IMAGE_DIR / "cost-accuracy-models.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
