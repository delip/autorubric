"""Plot few-shot learning results for RiceChem dataset."""

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_PATH = "autorubric-paper/images_paper1/fewshot_ricechem.pdf"

# Full data (do not discard for record-keeping)
shots_all = [0, 3, 5, 10, 20]
accuracy_all = [77.2, 79.0, 80.0, 79.7, 80.8]
cost_all = [0.51, 0.77, 0.92, 0.84, 1.07]

# Data for 0, 3, 5 shots only
shot_idx = [0, 1, 2]
shots = [shots_all[i] for i in shot_idx]
accuracy = [accuracy_all[i] for i in shot_idx]
cost = [cost_all[i] for i in shot_idx]

sns.set_theme(style="white", font_scale=1.1)
fig, ax1 = plt.subplots(figsize=(5, 3.5))

color_acc = sns.color_palette("deep")[0]
color_cost = sns.color_palette("deep")[3]

bar_width = 0.6

# Accuracy: bar chart
bars = ax1.bar(shots, accuracy, color=color_acc, width=bar_width, label="Accuracy")
ax1.set_xlabel("Number of shots")
ax1.set_ylabel("Accuracy (%)", color=color_acc)
ax1.tick_params(axis="y", labelcolor=color_acc)
ax1.set_xticks(shots)
ax1.set_ylim(0, 83)  # Start y-axis at 0 as requested

# Annotate bars with accuracy values
for bar, value in zip(bars, accuracy):
    height = bar.get_height()
    ax1.annotate(
        f"{value:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=color_acc,
        fontweight="bold",
    )

# Cost: overlay line chart
ax2 = ax1.twinx()
ax2.plot(
    shots,
    cost,
    marker="s",
    color=color_cost,
    linewidth=2,
    markersize=8,
    linestyle="--",
    label="Cost",
)
ax2.set_ylabel("Cost ($)", color=color_cost)
ax2.tick_params(axis="y", labelcolor=color_cost)
ax2.set_ylim(0, 1.3)  # Start y-axis at 0 as well

# Remove top border from both axes
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
