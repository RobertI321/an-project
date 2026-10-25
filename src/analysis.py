"""Metrics, statistical tests and plots for the retrained winners and baselines."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, mannwhitneyu

from src.genome import GRID_BINS, decode, grid_cell

COLOR = {"GA": "tab:red", "MAP-Elites": "tab:blue", "Optuna TPE": "tab:green", "Random search": "tab:gray"}
MARKER = {"GA": "^", "MAP-Elites": "o", "Optuna TPE": "D", "Random search": "s"}
BASE_STYLE = ["--", ":", "-."]


def add_metrics(configs, curves, test, cfg):
    """curves: (networks, seeds, eval points), test: (networks, seeds). Averages over seeds."""
    x = np.arange(1, curves.shape[2] + 1) * cfg["eval_every"]       # images seen at each eval point
    cfg["_x"] = x
    i_at, i_early = np.searchsorted(x, cfg["acc_at"]), np.searchsorted(x, cfg["early_samples"])
    for c, cu, te in zip(configs, curves, test):
        c["curve"], c["test"] = cu.mean(0), te.mean()
        c["acc_at"] = cu[:, i_at].mean()
        c["early_auc"] = cu[:, :i_early + 1].mean()                 # mean val acc in the early phase
        for th in cfg["thresholds"]:                                # images to reach th (nan = never)
            hit = cu >= th
            c[f"to{th}"] = np.nanmean(np.where(hit.any(1), x[hit.argmax(1)], np.nan)) if hit.any() else np.nan


def add_diversity(runs, cfg):
    """H1b: put ALL networks each search trained into the same grid (not just the winner)."""
    for B in cfg["budgets"]:
        rs = [r for r in runs if r["budget"] == B]
        best = max(r["F"].max() for r in rs)
        for r in rs:
            arch = np.full((GRID_BINS, GRID_BINS), -np.inf)
            for g, f in zip(r["G"], r["F"]):
                a, b = grid_cell(decode(g))
                arch[a, b] = max(arch[a, b], f)
            r["qd_score"] = np.sum(arch[np.isfinite(arch)] - 0.1)    # 0.1 = chance accuracy
            r["good_cells"] = int(np.sum(arch >= best - cfg["good_cell_margin"]))


def report(configs, cfg):
    bases = [c for c in configs if c["method"] == "Baseline"]
    runs = [c for c in configs if c["method"] != "Baseline"]
    sel = lambda B=None, m=None, key=None: np.array(
        [r[key] for r in runs if B in (None, r["budget"]) and m in (None, r["method"])], float)
    ms = lambda v: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
    ths = cfg["thresholds"]

    out = ["Speed = images to reach each validation accuracy (* = runs that never did)",
           f"{'budget':<8}{'method':<15}" + "".join(f"{'to ' + format(t, '.0%'):>10}" for t in ths)
           + f"{'acc@' + str(cfg['acc_at']):>17}{'early AUC':>17}{'final test':>17}{'QD-score':>10}{'good cells':>11}"]

    def row(B, m, rs):
        speed = ""
        for t in ths:
            v = np.array([r[f"to{t}"] for r in rs], float)
            speed += f"{(f'{np.nanmean(v):.0f}' if np.isfinite(v).any() else 'never') + '*' * np.isnan(v).sum():>10}"
        div = (f"{np.mean([r['qd_score'] for r in rs]):>10.1f}{np.mean([r['good_cells'] for r in rs]):>11.1f}"
               if "qd_score" in rs[0] else "")
        return (f"{B:<8}{m:<15}{speed}{ms([r['acc_at'] for r in rs]):>17}"
                f"{ms([r['early_auc'] for r in rs]):>17}{ms([r['test'] for r in rs]):>17}{div}")

    for b in bases:
        out.append(row("-", b["name"], [b]))
    for B in cfg["budgets"]:
        for m in cfg["methods"]:
            out.append(row(B, m, [r for r in runs if r["budget"] == B and r["method"] == m]))

    def greater(a, b, label):  # one-sided Mann-Whitney U, one value per search run
        if len(a) > 1 and len(b) > 1:
            p = mannwhitneyu(a, b, alternative="greater").pvalue
            out.append(f"  {label:<55} p = {p:.4f} {'*' if p < .05 else ''}")

    short, long = cfg["budgets"][0], cfg["budgets"][-1]
    out.append("\nH1  winners learn faster than the baselines (sign test on early AUC):")
    for b in bases:
        for B in cfg["budgets"]:
            v = sel(B, key="early_auc")
            k = int(np.sum(v > b["early_auc"]))
            p = binomtest(k, len(v), .5, alternative="greater").pvalue
            out.append(f"  vs {b['name']:<10} budget {B:<7}: {k}/{len(v)} winners faster   p = {p:.4f}")
    out.append("H1  short-budget winners learn faster early than long-budget winners:")
    for m in [None] + cfg["methods"]:
        greater(sel(short, m, "early_auc"), sel(long, m, "early_auc"), f"{m or 'ALL'}: early AUC {short} > {long}")
    out.append("H2  ... but reach lower final accuracy:")
    for m in [None] + cfg["methods"]:
        greater(sel(long, m, "test"), sel(short, m, "test"), f"{m or 'ALL'}: test acc {long} > {short}")
    out.append("Search methods vs random search (early AUC):")
    for B in cfg["budgets"]:
        for m in [m for m in cfg["methods"] if m != "Random search"]:
            greater(sel(B, m, "early_auc"), sel(B, "Random search", "early_auc"), f"budget {B}: {m} > Random")
    out.append("H1b MAP-Elites finds more diverse good networks:")
    for B in cfg["budgets"]:
        greater(sel(B, "MAP-Elites", "qd_score"), sel(B, "Random search", "qd_score"),
                f"budget {B}: QD-score MAP-Elites > Random")
    return "\n".join(out)


def plot(configs, cfg):
    bases = [c for c in configs if c["method"] == "Baseline"]
    runs = [c for c in configs if c["method"] != "Baseline"]
    budgets, methods, x = cfg["budgets"], cfg["methods"], cfg["_x"]
    bcol = {B: plt.get_cmap("viridis")(i / max(1, len(budgets) - 1) * .85) for i, B in enumerate(budgets)}
    fig, ax = plt.subplots(2, 3, figsize=(20, 10.5))
    ax = ax.ravel()

    for B in budgets:                                                # (a) learning curves
        C = np.stack([r["curve"] for r in runs if r["budget"] == B])
        ax[0].plot(x, C.mean(0), color=bcol[B], lw=2, label=f"winners of {B}-image search")
        ax[0].fill_between(x, C.mean(0) - C.std(0), C.mean(0) + C.std(0), color=bcol[B], alpha=.15)
    for b, ls in zip(bases, BASE_STYLE):
        ax[0].plot(x, b["curve"], "k", ls=ls, lw=2, label=f"baseline {b['name']}")
    ax[0].axvline(cfg["early_samples"], color="gray", ls=":")
    ax[0].set(xscale="log", xlabel="training images (log)", ylabel="validation accuracy",
              title="(a) Learning curves of winners")
    ax[0].legend(fontsize=8)

    for k, m in enumerate(methods):                                  # (b) early AUC
        for i, B in enumerate(budgets):
            v = [r["early_auc"] for r in runs if r["budget"] == B and r["method"] == m]
            xs = i + (k - 1.5) * .18
            ax[1].scatter([xs] * len(v), v, color=COLOR[m], marker=MARKER[m], label=m if i == 0 else None)
            ax[1].hlines(np.mean(v), xs - .07, xs + .07, color=COLOR[m], lw=2)
    for b, ls in zip(bases, BASE_STYLE):
        ax[1].axhline(b["early_auc"], color="k", ls=ls, label=f"baseline {b['name']}")
    ax[1].set_xticks(range(len(budgets)), [str(B) for B in budgets])
    ax[1].set(xlabel="search budget (images)", ylabel="early AUC", title="(b) H1: early learning speed")
    ax[1].legend(fontsize=8)

    for r in runs:                                                   # (c) trade-off
        ax[2].scatter(r["early_auc"], r["test"], color=bcol[r["budget"]], marker=MARKER[r["method"]], s=50)
    for b in bases:
        ax[2].scatter(b["early_auc"], b["test"], marker="*", s=300, c="k")
        ax[2].annotate(b["name"], (b["early_auc"], b["test"]), fontsize=8)
    for B in budgets:
        ax[2].scatter([], [], color=bcol[B], label=f"{B}-image winners")
    ax[2].set(xlabel="early AUC (speed)", ylabel="final test accuracy", title="(c) H2: do fast learners pay?")
    ax[2].legend(fontsize=8)

    gens = np.arange(1, cfg["generations"] + 1) * cfg["cpg"]         # (d) search progress
    for m in methods:
        best = np.array([r["best"] for r in runs if r["budget"] == budgets[0] and r["method"] == m])
        ax[3].plot(gens, best.mean(0), color=COLOR[m], marker=MARKER[m], label=m)
    ax[3].set(xlabel="networks trained", ylabel="best fitness so far",
              title=f"(d) Better than random? (budget {budgets[0]})")
    ax[3].legend()

    w = .8 / len(methods)                                            # (e) QD-score
    for k, m in enumerate(methods):
        v = [[r["qd_score"] for r in runs if r["budget"] == B and r["method"] == m] for B in budgets]
        ax[4].bar(np.arange(len(budgets)) + (k - 1.5) * w, [np.mean(a) for a in v], w,
                  yerr=[np.std(a) for a in v], color=COLOR[m], label=m, capsize=3)
    ax[4].set_xticks(range(len(budgets)), [str(B) for B in budgets])
    ax[4].set(xlabel="search budget (images)", ylabel="QD-score", title="(e) H1b: diversity of good networks")
    ax[4].legend()

    ax[5].axis("off")
    fig.tight_layout()
    return fig
