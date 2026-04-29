"""Generate the figures referenced in refs/Deepmind_Cooling.md.

Currently produces:
  * refs/figs/bcooler_pipeline.png — BCOOLER's 7-step decision pipeline
    (read sensors -> sample -> constraint filter -> Q-ensemble score ->
     95/5 exploit-explore -> apply -> daily retrain)

Run from the repo root: ``python refs/build_paper_review_figs.py``.

This script is reproducible and stand-alone; it has no dependencies
beyond matplotlib and numpy. It deliberately doesn't depend on any
of the project's RL code so that it can be run by anyone re-checking
the paper review.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def _box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        edgecolor="black", facecolor=color, linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", linespacing=1.2)
    return (x, y, w, h)


def _arrow(ax, src, dst, color="black", style="-", lw=1.6, label=None,
           label_offset=(0, 0)):
    sx, sy, sw, sh = src
    tx, ty, tw, th = dst
    if abs(sx - tx) < 0.01:
        # Vertical arrow: bottom of src -> top of dst
        start = (sx, sy - sh / 2)
        end = (tx, ty + th / 2)
    else:
        # Diagonal: bottom of src -> top of dst, accept the slight angle
        start = (sx, sy - sh / 2)
        end = (tx, ty + th / 2)
    arr = FancyArrowPatch(start, end,
                          arrowstyle="-|>", mutation_scale=14,
                          color=color, linestyle=style, linewidth=lw)
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, style="italic", color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))


def _curved_loop(ax, src_xy, dst_xy, label, label_xy, color="#888"):
    """Curved dashed feedback arrow used for the daily-retrain loop."""
    arr = FancyArrowPatch(
        src_xy, dst_xy,
        arrowstyle="-|>", mutation_scale=14,
        connectionstyle="arc3,rad=-0.45",
        color=color, linestyle="--", linewidth=1.4,
    )
    ax.add_patch(arr)
    ax.text(*label_xy, label, ha="center", va="center",
            fontsize=10, style="italic", color=color)


def build_bcooler_pipeline():
    fig, ax = plt.subplots(figsize=(11, 12.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14.5)
    ax.axis("off")

    # Color palette
    SENSOR  = "#d6e6ff"   # blue (input)
    SAMPLE  = "#fff5cc"   # yellow
    FILTER  = "#ffd7b3"   # orange — safety
    SCORE   = "#dcc7ff"   # purple — Q ensemble
    EXPLOIT = "#cdf2cd"   # green
    EXPLORE = "#ffcccc"   # red
    APPLY   = "#d6e6ff"   # blue (back to env)
    UPDATE  = "#dddddd"   # grey

    # Layout (top to bottom)
    sensor  = _box(ax, 6.0, 13.5, 5.5, 0.95,
                   "Read 50 sensor inputs (every 5 minutes)", SENSOR)
    sample  = _box(ax, 6.0, 11.9, 5.5, 0.95,
                   "Sample ~100,000 candidate actions", SAMPLE)
    flt     = _box(ax, 6.0, 10.3, 5.5, 1.10,
                   "Filter by hard safety constraints\n"
                   "(59 action constraints, 24 observation constraints)", FILTER,
                   fontsize=10)
    score   = _box(ax, 6.0, 8.6, 5.5, 1.10,
                   "Q-ensemble (10 networks)\n"
                   "predicts 15-minute energy use per candidate", SCORE,
                   fontsize=10)
    exploit = _box(ax, 3.2, 6.6, 4.5, 1.20,
                   "95%  EXPLOIT\n"
                   "lowest mean predicted energy\n"
                   "− penalty for high ensemble σ", EXPLOIT, fontsize=9)
    explore = _box(ax, 8.8, 6.6, 4.5, 1.20,
                   "5%  EXPLORE\n"
                   "highest ensemble σ\n"
                   "(novel state-action pairs)", EXPLORE, fontsize=9)
    apply_  = _box(ax, 6.0, 4.4, 5.5, 1.05,
                   "Apply chosen action to chiller plant\n"
                   "Observe new state 5 minutes later", APPLY, fontsize=10)
    update  = _box(ax, 6.0, 2.5, 5.5, 1.05,
                   "Daily: append transitions, retrain\n"
                   "Q-ensemble offline (10 networks)", UPDATE, fontsize=10)

    # Forward arrows
    _arrow(ax, sensor,  sample)
    _arrow(ax, sample,  flt)
    _arrow(ax, flt,     score)
    _arrow(ax, score,   exploit)
    _arrow(ax, score,   explore)
    _arrow(ax, exploit, apply_)
    _arrow(ax, explore, apply_)
    _arrow(ax, apply_,  update)

    # Feedback loop: update box -> back into score box (Q ensemble updated)
    _curved_loop(ax,
                 src_xy=(8.75, 2.5),
                 dst_xy=(8.75, 8.6),
                 label="updated weights\n(once per day)",
                 label_xy=(11.05, 5.4),
                 color="#666")

    # Title
    ax.text(6.0, 14.2, "BCOOLER decision pipeline (every 5 minutes)",
            ha="center", va="center", fontsize=15, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "bcooler_pipeline.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    build_bcooler_pipeline()
    print("Done.")
