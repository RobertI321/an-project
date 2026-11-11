"""Turn the raw results into tables (pandas) and statistical tests."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

ALPHA = 0.05
CHANCE = 0.1          # accuracy of random guessing (10 classes)


def images_to_reach(curve, threshold, eval_every):
    """Training images seen when the learning curve first reaches `threshold` (NaN = never)."""
    hits = np.flatnonzero(curve >= threshold)
    return (hits[0] + 1) * eval_every if len(hits) else np.nan


def network_table(networks, cfg):
    """One row per retrained network: its hyperparameters, speed and accuracy (averaged over retraining seeds).

    speed_to_XX%  images needed to reach XX% val accuracy (average of the seeds that got there; empty = none did)
    acc_at        val accuracy after `acc_at` images
    early_auc     average val accuracy over the first `early_images` images (one number for early speed)
    final_val     val accuracy at the end of retraining
    test_acc      test accuracy at the end of retraining
    """
    every, a = cfg["retrain"]["eval_every"], cfg["analysis"]
    rows = []
    for net in networks:
        curves = net["seed_curves"]
        row = {"name": net["name"], "method": net["method"], "budget": net["budget"], "seed": net["seed"], **net["hp"]}
        for th in a["thresholds"]:
            images = np.array([images_to_reach(c, th, every) for c in curves], dtype=float)
            row[f"speed_to_{th:.0%}"] = np.nanmean(images) if np.isfinite(images).any() else np.nan
        row["acc_at"] = curves[:, a["acc_at"] // every - 1].mean()
        row["early_auc"] = curves[:, :a["early_images"] // every].mean()
        row["final_val"] = curves[:, -1].mean()
        row["test_acc"] = net["test_accuracy"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["budget"] = df["budget"].astype("Int64")
    df["seed"] = df["seed"].astype("Int64")
    return df


def diversity_table(runs, space, cfg):
    """H1b: put ALL networks each search trained into the same grid (not only the winner).
    coverage = share of grid cells used, qd_score = sum over used cells of (best accuracy - chance),
    good_cells = cells with a network within `good_cell_margin` of the best accuracy at that budget."""
    best_per_budget = {b: max(r["fitness"].max() for r in runs if r["budget"] == b) for b in cfg["run"]["budgets"]}
    rows = []
    for run in runs:
        grid = np.full((space.grid_bins, space.grid_bins), np.nan)
        for genome, score in zip(run["genomes"], run["fitness"]):
            row, col = space.grid_cell(space.decode(genome))
            grid[row, col] = np.fmax(grid[row, col], score)
        used = grid[~np.isnan(grid)]
        rows.append({"budget": run["budget"], "method": run["method"], "seed": run["seed"],
                     "coverage": len(used) / grid.size,
                     "qd_score": np.sum(used - CHANCE),
                     "good_cells": int(np.sum(used >= best_per_budget[run["budget"]] - cfg["analysis"]["good_cell_margin"]))})
    return pd.DataFrame(rows)


def summary_table(df, diversity):
    """Averages of the winners per (budget, method), with the diversity columns added."""
    metrics = [c for c in df.columns if c.startswith("speed_to")] + ["acc_at", "early_auc", "final_val", "test_acc"]
    winners = df[df["method"] != "Baseline"].groupby(["budget", "method"])[metrics].mean()
    div = diversity.groupby(["budget", "method"])[["coverage", "qd_score", "good_cells"]].mean()
    return winners.join(div)


def hypothesis_tests(df, diversity, cfg):
    """All statistical tests in one table. One data point = one search run (one winner)."""
    winners, baselines = df[df["method"] != "Baseline"], df[df["method"] == "Baseline"]
    budgets, methods = cfg["run"]["budgets"], cfg["search"]["methods"]
    short, long = budgets[0], budgets[-1]
    rows = []

    def add(hypothesis, comparison, result, p):
        rows.append({"hypothesis": hypothesis, "comparison": comparison, "result": result,
                     "p": p, "significant": bool(p < ALPHA)})

    def greater(hypothesis, comparison, a, b):
        """One-sided Mann-Whitney U: are the values in a larger than in b?"""
        if len(a) >= 2 and len(b) >= 2:
            add(hypothesis, comparison, f"{np.mean(a):.4f} vs {np.mean(b):.4f}",
                mannwhitneyu(a, b, alternative="greater").pvalue)

    # H1 (a): winners are faster early than the best baseline -> sign test: how many of them beat it?
    best_baseline = baselines["early_auc"].max()
    for label, group in [("all budgets", winners)] + [(f"budget {b}", winners[winners["budget"] == b]) for b in budgets]:
        k, n = int((group["early_auc"] > best_baseline).sum()), len(group)
        add("H1 faster than best baseline", label, f"{k}/{n} winners faster",
            binomtest(k, n, 0.5, alternative="greater").pvalue)

    if short != long:
        sel = lambda b, m=None: winners[(winners["budget"] == b) & ((winners["method"] == m) if m else True)]
        # H1 (b): short-budget winners are faster early than long-budget winners
        for m in [None] + methods:
            greater("H1 short budget faster early", f"{m or 'all methods'}: {short} > {long}",
                    sel(short, m)["early_auc"], sel(long, m)["early_auc"])
        # H2: ... but end with lower final accuracy
        for m in [None] + methods:
            greater("H2 long budget better at the end", f"{m or 'all methods'}: {long} > {short}",
                    sel(long, m)["test_acc"], sel(short, m)["test_acc"])

    # Search methods vs random search (early speed of their winners)
    for b in budgets:
        at_b = winners[winners["budget"] == b]
        for m in [m for m in methods if m != "Random search"]:
            greater("Method beats random search", f"budget {b}: {m}",
                    at_b[at_b["method"] == m]["early_auc"], at_b[at_b["method"] == "Random search"]["early_auc"])

    # H1b: MAP-Elites finds more diverse good networks
    for b in budgets:
        at_b = diversity[diversity["budget"] == b]
        greater("H1b MAP-Elites more diverse", f"budget {b}: QD-score",
                at_b[at_b["method"] == "MAP-Elites"]["qd_score"], at_b[at_b["method"] == "Random search"]["qd_score"])

    return pd.DataFrame(rows)
