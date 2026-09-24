from pathlib import Path
import json
import argparse
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    description="Render the risk curves and audit-score comparison"
)
parser.add_argument("--output", type=Path, default=R / "generated")
args = parser.parse_args()
A = args.output
A.mkdir(parents=True, exist_ok=True)
DATA = json.loads((R / "figure_data.json").read_text())
D = DATA["curves"]
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9.2,
        "axes.labelsize": 9.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axs = plt.subplots(1, 2, figsize=(7.008, 1.82))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.26, top=0.84, wspace=0.36)
cols = ["#385d73", "#93714d", "#5d7664"]
for i, (seed, row) in enumerate(D["seeds"].items()):
    axs[0].plot(
        row["coverage"],
        row["risk_curve"],
        color=cols[i],
        ls=["-", "--", "-."][i],
        lw=1.3,
        label=f"Run {i + 1}",
    )
axs[0].plot(
    D["seeds"]["2027"]["coverage"],
    D["soft_dice"]["risk_curve"],
    ":",
    color="#272c32",
    lw=1.7,
    label="Soft Dice",
)
axs[0].set(
    xlim=(0.05, 1), ylim=(0.38, 0.82), xlabel="Coverage", ylabel="Retained-mask risk"
)
axs[0].set_title("(a) Final dual across runs", loc="left", pad=7)
axs[0].legend(
    loc="lower right", ncol=2, columnspacing=0.65, handlelength=1.6, frameon=False
)
axs[0].set_xticks([0.2, 0.4, 0.6, 0.8, 1])
offsets = {
    "Pooled": ((5, -15), "s"),
    "w/o gap difference": ((0, 5), "^"),
    "Dual (16 ep.)": ((7, -4), "D"),
    "CE": ((-20, 5), "v"),
    "Final dual": ((5, 5), "*"),
}
pts = [
    (p["label"], p["aurc"], p["invalid_audio_auroc"], *offsets[p["label"]])
    for p in DATA["points"]
]
for label, x, y, offset, marker in pts:
    axs[1].scatter(
        x,
        y,
        marker=marker,
        s=65 if marker == "*" else 26,
        c="#385d73" if marker == "*" else "#806b53",
        zorder=3,
    )
    axs[1].annotate(
        label, (x, y), xytext=offset, textcoords="offset points", fontsize=9.2
    )
axs[1].set(
    xlim=(0.618, 0.670),
    ylim=(0.850, 0.935),
    xlabel="AURC (lower is better)",
    ylabel="Invalid-audio AUROC",
)
axs[1].set_title("(b) Complementary audit scores", loc="left", pad=7)
axs[1].set_xticks([0.62, 0.64, 0.66])
for ax in axs:
    ax.grid(alpha=0.16, lw=0.5)
for ext in ("pdf", "svg", "png"):
    fig.savefig(A / f"set_aware_reliability_A.{ext}", dpi=300)
plt.close(fig)
