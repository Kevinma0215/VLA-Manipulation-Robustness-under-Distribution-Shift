"""
experiments/analysis.py

Analyse batch evaluation results from experiments/results.csv.
Produces 4 plots saved to experiments/plots/ and prints a summary table.

Usage:
    python experiments/analysis.py
"""

import os
import sys
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_CSV  = os.path.join(os.path.dirname(__file__), "results.csv")
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "plots")
DPI          = 150
FONT_SIZE    = 12

CONDITION_ORDER  = ["nominal", "mild", "medium", "strong"]
N_EPISODES       = 20
OFFSET_LABELS    = {"nominal": "0 cm", "mild": "5 cm",
                    "medium": "10 cm", "strong": "15 cm"}
OFFSET_VALUES    = {"nominal": 0.00, "mild": -0.05,
                    "medium": -0.10, "strong": -0.15}

FAILURE_TYPES    = ["no_grasp", "drop", "wrong_place"]
FAILURE_COLORS   = {
    "success":     "#2ca02c",   # green
    "no_grasp":    "#d62728",   # red
    "drop":        "#ff7f0e",   # orange
    "wrong_place": "#9467bd",   # purple
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def wilson_ci(n_success: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for a proportion. Returns (lo, hi) in %."""
    if n == 0:
        return 0.0, 0.0
    p = n_success / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100)


def load_data() -> pd.DataFrame | None:
    if not os.path.exists(RESULTS_CSV):
        print(f"[analysis] ERROR: {RESULTS_CSV} not found. Run eval_runner.py first.")
        return None
    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("[analysis] ERROR: results.csv is empty.")
        return None

    # normalise types
    df["success"]        = df["success"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}).fillna(False)
    df["episode_length"] = pd.to_numeric(df["episode_length"], errors="coerce")
    df["condition"]      = df["condition"].astype(str)

    # ── deduplication: keep last row per (condition, seed) ───────────────────
    for cond in df["condition"].unique():
        n_before = (df["condition"] == cond).sum()
        if n_before > N_EPISODES:
            n_drop = n_before - N_EPISODES
            print(
                f"[analysis] WARNING: {cond} had {n_before} rows, "
                f"dropped {n_drop} duplicates, kept {N_EPISODES}"
            )
    df = df.groupby(["condition", "seed"], sort=False).last().reset_index()

    present = df["condition"].unique().tolist()
    missing = [c for c in CONDITION_ORDER if c not in present]
    if missing:
        warnings.warn(f"[analysis] Missing conditions (partial data): {missing}")
        print(f"[analysis] WARNING: conditions not yet in CSV: {missing}")

    # ── warn on incomplete conditions ─────────────────────────────────────────
    for cond in present:
        n = (df["condition"] == cond).sum()
        if n < N_EPISODES:
            print(
                f"[analysis] WARNING: {cond} only has {n}/{N_EPISODES} episodes "
                f"complete — results may be noisy"
            )

    # derive a clean label column in correct order
    df["cond_order"] = df["condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)})
    df = df.sort_values("cond_order")

    return df


def per_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-condition summary DataFrame."""
    rows = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        n         = len(sub)
        n_success = sub["success"].sum()
        sr        = n_success / n * 100
        lo, hi    = wilson_ci(n_success, n)
        avg_len   = sub["episode_length"].mean()
        failures  = sub[~sub["success"]]["failure_type"].value_counts()
        top_fail  = failures.idxmax() if not failures.empty else "-"
        counts    = {ft: int((sub["failure_type"] == ft).sum()) for ft in FAILURE_TYPES}

        rows.append({
            "condition":   cond,
            "offset_label": OFFSET_LABELS[cond],
            "offset_cm":   abs(OFFSET_VALUES[cond]) * 100,
            "n":           n,
            "n_success":   int(n_success),
            "success_pct": sr,
            "wilson_lo":   lo,
            "wilson_hi":   hi,
            "avg_length":  avg_len,
            "top_failure": top_fail,
            **counts,
        })
    return pd.DataFrame(rows)


def apply_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "font.size":        FONT_SIZE,
        "axes.titlesize":   FONT_SIZE + 2,
        "axes.labelsize":   FONT_SIZE,
        "xtick.labelsize":  FONT_SIZE,
        "ytick.labelsize":  FONT_SIZE,
        "legend.fontsize":  FONT_SIZE - 1,
    })


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — success rate
# ──────────────────────────────────────────────────────────────────────────────

def plot_success_rate(stats: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x      = stats["offset_cm"].values
    y      = stats["success_pct"].values
    lo     = y - stats["wilson_lo"].values
    hi     = stats["wilson_hi"].values - y
    labels = stats["condition"].values

    ax.errorbar(x, y, yerr=[lo, hi],
                fmt="o-", color="#1f77b4", linewidth=2, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5, label="Success rate")

    # nominal reference line
    nominal_row = stats[stats["condition"] == "nominal"]
    if not nominal_row.empty:
        nom_sr = nominal_row["success_pct"].values[0]
        ax.axhline(nom_sr, color="#1f77b4", linestyle="--", linewidth=1.2,
                   alpha=0.55, label=f"Nominal ({nom_sr:.0f}%)")

    ax.set_xlabel("Initial EE Offset Magnitude (cm)", fontsize=FONT_SIZE)
    ax.set_ylabel("Success Rate (%)", fontsize=FONT_SIZE)
    ax.set_title("Success Rate vs Initial EE Offset", fontsize=FONT_SIZE + 2)
    ax.set_ylim(-5, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi:.0f}" for xi in x])

    # secondary x-axis with condition names
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=FONT_SIZE - 1)
    ax2.tick_params(axis="x", length=0)

    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — failure breakdown (stacked bar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_failure_breakdown(stats: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    segments = ["success"] + FAILURE_TYPES
    x        = np.arange(len(stats))
    bottoms  = np.zeros(len(stats))

    for seg in segments:
        if seg == "success":
            heights = stats["n_success"].values.astype(float)
        else:
            heights = stats[seg].values.astype(float)
        ax.bar(x, heights, bottom=bottoms,
               color=FAILURE_COLORS[seg], label=seg.replace("_", " "),
               edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(stats["condition"].tolist(), fontsize=FONT_SIZE)
    ax.set_ylabel("Episode Count", fontsize=FONT_SIZE)
    ax.set_xlabel("Condition", fontsize=FONT_SIZE)
    ax.set_title("Failure Mode Distribution by Condition", fontsize=FONT_SIZE + 2)
    ax.set_ylim(0, max(stats["n"].max() * 1.1, 22))

    # episode count on top of each bar
    for i, row in stats.iterrows():
        idx = stats.index.get_loc(i)
        ax.text(idx, row["n"] + 0.3, str(int(row["n"])),
                ha="center", va="bottom", fontsize=FONT_SIZE - 1)

    handles = [mpatches.Patch(color=FAILURE_COLORS[s],
               label=s.replace("_", " ")) for s in segments]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9, fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 — episode length box + strip
# ──────────────────────────────────────────────────────────────────────────────

def plot_episode_length(df: pd.DataFrame, out_path: str):
    present = [c for c in CONDITION_ORDER if c in df["condition"].values]
    data    = [df[df["condition"] == c]["episode_length"].dropna().values
               for c in present]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2},
                    boxprops={"facecolor": "#aec6e8", "alpha": 0.75},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "", "markersize": 0})

    # jittered strip
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   alpha=0.55, s=28, color="#1f77b4", zorder=3)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, fontsize=FONT_SIZE)
    ax.set_xlabel("Condition", fontsize=FONT_SIZE)
    ax.set_ylabel("Episode Length (steps)", fontsize=FONT_SIZE)
    ax.set_title("Episode Length Distribution by Condition", fontsize=FONT_SIZE + 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4 — degradation summary (hero figure)
# ──────────────────────────────────────────────────────────────────────────────

def plot_degradation_summary(stats: pd.DataFrame, out_path: str):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.38})

    x      = stats["offset_cm"].values
    labels = stats["condition"].values

    # ── top: success rate curve ──────────────────────────────────────────────
    y  = stats["success_pct"].values
    lo = y - stats["wilson_lo"].values
    hi = stats["wilson_hi"].values - y
    ax_top.errorbar(x, y, yerr=[lo, hi],
                    fmt="o-", color="#1f77b4", linewidth=2, markersize=8,
                    capsize=5, capthick=1.5, elinewidth=1.5)
    nominal_row = stats[stats["condition"] == "nominal"]
    if not nominal_row.empty:
        nom_sr = nominal_row["success_pct"].values[0]
        ax_top.axhline(nom_sr, color="#1f77b4", linestyle="--",
                       linewidth=1.2, alpha=0.55, label=f"Nominal ({nom_sr:.0f}%)")
        ax_top.legend(framealpha=0.9, fontsize=FONT_SIZE - 1)

    ax_top.set_ylabel("Success Rate (%)", fontsize=FONT_SIZE)
    ax_top.set_title("Success Rate vs Initial EE Offset", fontsize=FONT_SIZE + 2)
    ax_top.set_ylim(-5, 105)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{xi:.0f} cm\n({lbl})" for xi, lbl in zip(x, labels)],
                            fontsize=FONT_SIZE - 1)

    # ── bottom: normalised stacked failure % ────────────────────────────────
    segments = ["success"] + FAILURE_TYPES
    bottoms  = np.zeros(len(stats))
    xi       = np.arange(len(stats))

    for seg in segments:
        if seg == "success":
            raw = stats["n_success"].values.astype(float)
        else:
            raw = stats[seg].values.astype(float)
        heights = raw / stats["n"].values * 100
        ax_bot.bar(xi, heights, bottom=bottoms,
                   color=FAILURE_COLORS[seg], label=seg.replace("_", " "),
                   edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax_bot.set_xticks(xi)
    ax_bot.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax_bot.set_ylabel("Episode Share (%)", fontsize=FONT_SIZE)
    ax_bot.set_xlabel("Condition", fontsize=FONT_SIZE)
    ax_bot.set_title("Failure Mode Breakdown (Normalised)", fontsize=FONT_SIZE + 2)
    ax_bot.set_ylim(0, 108)

    handles = [mpatches.Patch(color=FAILURE_COLORS[s],
               label=s.replace("_", " ")) for s in segments]
    ax_bot.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  framealpha=0.9, fontsize=FONT_SIZE - 1)

    fig.suptitle("SmolVLA Pick-and-Place: EE Offset Degradation",
                 fontsize=FONT_SIZE + 3, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(stats: pd.DataFrame):
    col_w = 78
    print()
    print("=" * col_w)
    print("  EVALUATION SUMMARY")
    print("=" * col_w)
    header = (f"{'Condition':<10} | {'Offset':>7} | {'N':>3} | "
              f"{'Successes':>9} | {'Success%':>8} | "
              f"{'Avg Len':>7} | {'Top Failure':<14}")
    print(header)
    print("-" * col_w)
    for _, row in stats.iterrows():
        sr_str  = f"{row['success_pct']:.1f}%"
        len_str = f"{row['avg_length']:.0f}" if not math.isnan(row["avg_length"]) else "—"
        print(
            f"{row['condition']:<10} | "
            f"{row['offset_cm']:>5.0f} cm | "
            f"{row['n']:>3} | "
            f"{row['n_success']:>9} | "
            f"{sr_str:>8} | "
            f"{len_str:>7} | "
            f"{row['top_failure']:<14}"
        )
    print("=" * col_w)

    # failure counts
    print()
    print("-" * col_w)
    header2 = (f"{'Condition':<10} | {'no_grasp':>9} | {'drop':>6} | "
               f"{'timeout':>8} | {'wrong_place':>12} | {'total_fail':>10}")
    print(header2)
    print("-" * col_w)
    for _, row in stats.iterrows():
        total = row["n"] - row["n_success"]
        print(
            f"{row['condition']:<10} | "
            f"{row['no_grasp']:>9} | "
            f"{row['drop']:>6} | "
            f"{row['timeout']:>8} | "
            f"{row['wrong_place']:>12} | "
            f"{int(total):>10}"
        )
    print("=" * col_w)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    apply_style()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    if df is None:
        sys.exit(1)

    stats = per_condition_stats(df)
    if stats.empty:
        print("[analysis] No valid conditions found in results.csv.")
        sys.exit(1)

    print(f"[analysis] Loaded {len(df)} rows from {RESULTS_CSV}")
    print(f"[analysis] Conditions present: {stats['condition'].tolist()}")
    print(f"[analysis] Saving plots to {PLOTS_DIR}/\n")

    print_summary_table(stats)

    print("Generating plots...")
    plot_success_rate(
        stats, os.path.join(PLOTS_DIR, "success_rate.png"))
    plot_failure_breakdown(
        stats, os.path.join(PLOTS_DIR, "failure_breakdown.png"))
    plot_episode_length(
        df,    os.path.join(PLOTS_DIR, "episode_length.png"))
    plot_degradation_summary(
        stats, os.path.join(PLOTS_DIR, "degradation_summary.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
