"""Generate the length penalty curves chart for the cookbook."""

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

FREE_BUDGET = 200
MAX_CAP = 400
PENALTY_AT_CAP = 0.30

EXPONENTS = [
    (1.0, "Linear (exp=1.0)", "--", 1.2),
    (1.6, "Default (exp=1.6)", "-", 2.0),
    (2.5, "Steep (exp=2.5)", ":", 1.2),
]

COLORS = ["#e76f51", "#2a9d8f", "#264653"]


def compute_penalty(counts: np.ndarray, exponent: float) -> np.ndarray:
    penalty = np.zeros_like(counts)
    mid = (counts > FREE_BUDGET) & (counts < MAX_CAP)
    ratio = (counts[mid] - FREE_BUDGET) / (MAX_CAP - FREE_BUDGET)
    penalty[mid] = PENALTY_AT_CAP * (ratio**exponent)
    penalty[counts >= MAX_CAP] = PENALTY_AT_CAP
    return penalty * 100


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    counts = np.linspace(0, 500, 300)

    with THEME:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Shaded zones
        ax.axvspan(0, FREE_BUDGET, color="#d0e4f7", alpha=0.35, label="Free budget")
        ax.axvspan(MAX_CAP, 500, color="#f7d0d0", alpha=0.35, label="Capped")

        for (exp, label, ls, lw), color in zip(EXPONENTS, COLORS):
            penalty = compute_penalty(counts, exp)
            ax.plot(counts, penalty, linestyle=ls, linewidth=lw, color=color, label=label)

        ax.set_xlabel("Word count")
        ax.set_ylabel("Penalty (%)")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 35)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7)
        fig.suptitle("Length Penalty Curves by Exponent", fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.85)

        out = IMAGE_DIR / "length-penalty-curve.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
