"""Generate charts for the agent skill evaluation cookbook."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme

DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "peer_review_skill_eval.json"
IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"

THEME = (
    load_theme("scientific")
    .set_ticks(size_major=4, size_minor=2, width_major=0.8, width_minor=0.5)
    .set_grid(draw=False)
    .set_font(family="sans-serif", size=8)
    .set_axis_labels(pad=8)
)

COLORS = {"without": "#9e9e9e", "poor": "#f4a261", "good": "#2a9d8f"}
LABELS = {"without-skill": "Without Skill", "poor-skill": "Poor Skill", "good-skill": "Good Skill"}
CONDITION_KEYS = ["without-skill", "poor-skill", "good-skill"]
COLOR_LIST = [COLORS["without"], COLORS["poor"], COLORS["good"]]

DIMENSIONS = {
    "Outcome": [
        "paper_summary",
        "methodology_assessment",
        "statistical_evaluation",
        "strengths_and_weaknesses",
        "clear_recommendation",
    ],
    "Style": ["constructive_tone", "structured_format", "specific_references"],
    "Efficiency": ["concise_review"],
}


def load_data() -> tuple[list[str], dict[str, list[list[str]]], list[float]]:
    """Load dataset and return (criteria_names, conditions_verdicts, weights)."""
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    criteria = [c["name"] for c in data["rubric"]]
    weights = [c["weight"] for c in data["rubric"]]
    conditions: dict[str, list[list[str]]] = {k: [] for k in CONDITION_KEYS}

    for item in data["items"]:
        desc = item["description"]
        for cond in conditions:
            if f"[{cond}]" in desc:
                conditions[cond].append(item["ground_truth"])
                break

    return criteria, conditions, weights


def compute_pass_rates(
    criteria: list[str], conditions: dict[str, list[list[str]]]
) -> dict[str, list[float]]:
    """Compute per-criterion pass rate (%) for each condition."""
    rates: dict[str, list[float]] = {}
    for cond, verdicts_list in conditions.items():
        rates[cond] = []
        for i in range(len(criteria)):
            met = sum(1 for v in verdicts_list if v[i] == "MET")
            rates[cond].append(met / len(verdicts_list) * 100)
    return rates


def _save(fig: plt.Figure, name: str) -> None:
    out = IMAGE_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def chart_pass_rates(criteria: list[str], rates: dict[str, list[float]]) -> None:
    """Grouped bar chart: per-criterion pass rate by condition."""
    short_labels = [name.replace("_", " ") for name in criteria]
    x = np.arange(len(criteria))
    width = 0.25

    with THEME:
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (cond, color) in enumerate(zip(CONDITION_KEYS, COLOR_LIST)):
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset, rates[cond], width, label=LABELS[cond], color=color, edgecolor="0.95"
            )
            for bar, val in zip(bars, rates[cond]):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{val:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=5,
                    )

        ax.set_ylabel("Pass rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        ax.set_ylim(0, 115)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
        fig.suptitle("Per-Criterion Pass Rate by Condition", fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.85)

        _save(fig, "skill-eval-pass-rates.png")


def chart_overall_scores(conditions: dict[str, list[list[str]]], weights: list[float]) -> None:
    """Horizontal bar chart: overall weighted score per condition."""
    pos_weight = sum(w for w in weights if w > 0)
    scores = {}
    for cond, verdicts_list in conditions.items():
        item_scores = []
        for verdicts in verdicts_list:
            weighted_sum = sum(w for v, w in zip(verdicts, weights) if v == "MET")
            item_scores.append(max(0.0, min(1.0, weighted_sum / pos_weight)))
        scores[cond] = sum(item_scores) / len(item_scores)

    vals = [scores[c] for c in CONDITION_KEYS]

    with THEME:
        fig, ax = plt.subplots(figsize=(6, 2.8))

        bars = ax.barh(
            range(len(CONDITION_KEYS)), vals, color=COLOR_LIST, edgecolor="0.95", height=0.55
        )

        for bar, val in zip(bars, vals):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

        # Delta annotations between bars
        for i in range(1, len(vals)):
            delta = vals[i] - vals[i - 1]
            mid_x = (vals[i - 1] + vals[i]) / 2
            mid_y = i - 0.5
            ax.annotate(
                f"+{delta:.2f}",
                xy=(mid_x, mid_y),
                fontsize=7,
                fontweight="bold",
                color="#555555",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9
                ),
            )

        ax.set_yticks(range(len(CONDITION_KEYS)))
        ax.set_yticklabels([LABELS[c] for c in CONDITION_KEYS])
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Weighted score")
        ax.set_title("Overall Score by Condition", fontweight="bold")
        ax.invert_yaxis()

        _save(fig, "skill-eval-overall-scores.png")


def chart_lift(criteria: list[str], rates: dict[str, list[float]]) -> None:
    """Horizontal bar chart: per-criterion delta (good-skill minus without-skill), sorted."""
    deltas = []
    for i, name in enumerate(criteria):
        delta = rates["good-skill"][i] - rates["without-skill"][i]
        deltas.append((name, delta))

    deltas.sort(key=lambda x: x[1])
    names = [d[0].replace("_", " ").title() for d in deltas]
    values = [d[1] for d in deltas]
    bar_colors = [COLORS["good"] if v >= 0 else "#e76f51" for v in values]

    with THEME:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        bars = ax.barh(range(len(names)), values, color=bar_colors, edgecolor="0.95", height=0.6)

        for bar, val in zip(bars, values):
            if val >= 0:
                ax.text(
                    val + 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.0f}",
                    va="center",
                    ha="left",
                    fontsize=6,
                    fontweight="bold",
                )
            else:
                # Place inside the bar to avoid overlap with y-axis labels
                ax.text(
                    val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.0f}",
                    va="center",
                    ha="center",
                    fontsize=6,
                    fontweight="bold",
                    color="white",
                )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Pass rate change (pp)")
        ax.set_title("Skill Lift: Good Skill vs. No Skill", fontweight="bold")
        ax.axvline(0, color="0.3", linewidth=0.6)

        _save(fig, "skill-eval-lift.png")


def chart_dimensions(criteria: list[str], conditions: dict[str, list[list[str]]]) -> None:
    """Grouped bar chart: dimension-level pass rates across conditions."""
    dim_rates: dict[str, list[float]] = {}
    for dim_name, crit_names in DIMENSIONS.items():
        dim_rates[dim_name] = []
        for cond in CONDITION_KEYS:
            met = 0
            total = 0
            for i, name in enumerate(criteria):
                if name in crit_names:
                    met += sum(1 for v in conditions[cond] if v[i] == "MET")
                    total += len(conditions[cond])
            dim_rates[dim_name].append(met / total * 100 if total > 0 else 0)

    dim_names = list(DIMENSIONS.keys())
    x = np.arange(len(dim_names))
    width = 0.25

    with THEME:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for i, (cond, color) in enumerate(zip(CONDITION_KEYS, COLOR_LIST)):
            vals = [dim_rates[d][i] for d in dim_names]
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset, vals, width, label=LABELS[cond], color=color, edgecolor="0.95"
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        ax.set_ylabel("Pass rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.set_ylim(0, 118)
        ax.legend(loc="upper left", ncol=1, frameon=False)
        fig.suptitle("Dimension-Level Pass Rates by Condition", fontweight="bold")
        fig.subplots_adjust(top=0.90)

        _save(fig, "skill-eval-dimensions.png")


def chart_heatmap(criteria: list[str], rates: dict[str, list[float]]) -> None:
    """Criterion x condition heatmap with pass rate annotations."""
    matrix = np.array([rates[c] for c in CONDITION_KEYS]).T  # (criteria, conditions)
    col_labels = [LABELS[c] for c in CONDITION_KEYS]
    row_labels = [name.replace("_", " ").title() for name in criteria]

    with THEME:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

        for i in range(len(criteria)):
            for j in range(len(CONDITION_KEYS)):
                val = matrix[i, j]
                text_color = "white" if val > 60 else "0.2"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title("Pass Rate Heatmap (%)", fontweight="bold", pad=10)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Cell border grid (content-specific, not theme-level)
        ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        _save(fig, "skill-eval-heatmap.png")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    criteria, conditions, weights = load_data()
    rates = compute_pass_rates(criteria, conditions)

    chart_pass_rates(criteria, rates)
    chart_overall_scores(conditions, weights)
    chart_lift(criteria, rates)
    chart_dimensions(criteria, conditions)
    chart_heatmap(criteria, rates)


if __name__ == "__main__":
    main()
