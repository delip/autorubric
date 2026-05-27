"""Generate charts for the skill improvement cookbook recipe.

Loads results from examples/data/skill_improvement_results.json and generates:
  1. skill-improvement-convergence.png -- mean score vs iteration + gold baseline
  2. skill-improvement-before-after.png -- v1 vs improved per-criterion pass rates
  3. skill-improvement-three-way.png -- v1 vs improved vs gold per-criterion pass rates
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme

DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "skill_improvement_results.json"
IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"

THEME = (
    load_theme("scientific")
    .set_ticks(size_major=4, size_minor=2, width_major=0.8, width_minor=0.5)
    .set_grid(draw=False)
    .set_font(family="sans-serif", size=8)
    .set_axis_labels(pad=8)
)

COLOR_CORAL = "#e76f51"
COLOR_TEAL = "#2a9d8f"
COLOR_DARK = "#264653"
COLOR_AMBER = "#f4a261"
COLOR_GRAY = "#9e9e9e"


def load_data() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str) -> None:
    out = IMAGE_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def chart_convergence(data: dict) -> None:
    """Line chart: mean score vs iteration, with curated skill dashed baseline."""
    iterations = data["iterations"]
    iters = [it["iteration"] for it in iterations]
    scores = [it["mean_score"] for it in iterations]
    gold_score = data["gold_comparison"]["mean_score"]

    with THEME:
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(
            iters,
            scores,
            marker="o",
            color=COLOR_TEAL,
            linewidth=1.8,
            markersize=5,
            label="Improved skill",
            zorder=3,
        )

        for x, y in zip(iters, scores):
            # Place labels below for points near the top to avoid clipping
            offset = -0.035 if y > 0.95 else 0.025
            va = "top" if y > 0.95 else "bottom"
            ax.text(x, y + offset, f"{y:.2f}", ha="center", va=va, fontsize=6)

        ax.axhline(
            gold_score,
            linestyle="--",
            color=COLOR_AMBER,
            linewidth=1.2,
            label=f"Curated skill ({gold_score:.2f})",
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean score")
        ax.set_xlim(-0.3, iters[-1] + 0.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(iters)

        _save(fig, "skill-improvement-convergence.png")


def chart_before_after(data: dict) -> None:
    """Grouped bars: v1 (coral) vs improved (teal) per-criterion pass rates."""
    iterations = data["iterations"]
    v1_rates = iterations[0]["pass_rates"]
    improved_rates = iterations[-1]["pass_rates"]
    criteria = list(v1_rates.keys())
    short_labels = [name.replace("_", " ") for name in criteria]

    v1_vals = [v1_rates[c] * 100 for c in criteria]
    improved_vals = [improved_rates[c] * 100 for c in criteria]

    x = np.arange(len(criteria))
    width = 0.35

    with THEME:
        fig, ax = plt.subplots(figsize=(8, 4))

        bars_v1 = ax.bar(
            x - width / 2,
            v1_vals,
            width,
            label="Preliminary Skill",
            color=COLOR_CORAL,
            edgecolor="0.95",
        )
        bars_improved = ax.bar(
            x + width / 2,
            improved_vals,
            width,
            label="Improved Skill",
            color=COLOR_TEAL,
            edgecolor="0.95",
        )

        for bar, val in zip(bars_v1, v1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )
        for bar, val in zip(bars_improved, improved_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

        ax.set_ylabel("Pass rate (%)")
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )

        _save(fig, "skill-improvement-before-after.png")


def chart_three_way(data: dict) -> None:
    """Grouped bars: v1 (gray) vs improved (amber) vs gold (teal) per-criterion."""
    iterations = data["iterations"]
    v1_rates = iterations[0]["pass_rates"]
    improved_rates = iterations[-1]["pass_rates"]
    gold_rates = data["gold_comparison"]["pass_rates"]
    criteria = list(v1_rates.keys())
    short_labels = [name.replace("_", " ") for name in criteria]

    v1_vals = [v1_rates[c] * 100 for c in criteria]
    improved_vals = [improved_rates[c] * 100 for c in criteria]
    gold_vals = [gold_rates[c] * 100 for c in criteria]

    x = np.arange(len(criteria))
    width = 0.25

    with THEME:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        bars_v1 = ax.bar(
            x - width,
            v1_vals,
            width,
            label="Preliminary Skill",
            color=COLOR_GRAY,
            edgecolor="0.95",
        )
        bars_improved = ax.bar(
            x,
            improved_vals,
            width,
            label="Improved Skill",
            color=COLOR_AMBER,
            edgecolor="0.95",
        )
        bars_gold = ax.bar(
            x + width,
            gold_vals,
            width,
            label="Curated Skill",
            color=COLOR_TEAL,
            edgecolor="0.95",
        )

        # Only annotate values that differ from 100 to reduce clutter
        for bars, vals in [
            (bars_v1, v1_vals),
            (bars_improved, improved_vals),
            (bars_gold, gold_vals),
        ]:
            for bar, val in zip(bars, vals):
                if val != 100:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 1) + 1,
                        f"{val:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=5,
                    )

        ax.set_ylabel("Pass rate (%)")
        ax.set_ylim(0, 118)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
        )

        _save(fig, "skill-improvement-three-way.png")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    chart_convergence(data)
    chart_before_after(data)
    chart_three_way(data)


if __name__ == "__main__":
    main()
