"""Figures, one function per plot. Each saves a png and closes the figure."""
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLOR = {"GA": "tab:red", "MAP-Elites": "tab:blue", "Optuna TPE": "tab:green", "Random search": "tab:gray"}
METHOD_MARKER = {"GA": "^", "MAP-Elites": "o", "Optuna TPE": "D", "Random search": "s"}
BASELINE_STYLE = ["--", ":", "-."]


def budget_colors(budgets):
    return dict(zip(budgets, plt.cm.viridis(np.linspace(0, 0.9, len(budgets)))))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def learning_curves(networks, cfg, path):
    """Average learning curve of the winners of each budget vs the baselines."""
    budgets, colors = cfg["run"]["budgets"], budget_colors(cfg["run"]["budgets"])
    images = np.arange(1, len(networks[0]["curve"]) + 1) * cfg["retrain"]["eval_every"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for b in budgets:
        curves = np.stack([n["curve"] for n in networks if n["budget"] == b])
        ax.plot(images, curves.mean(0), color=colors[b], lw=2, label=f"winners of {b:,}-image searches")
    for n, style in zip([n for n in networks if n["method"] == "Baseline"], BASELINE_STYLE):
        ax.plot(images, n["curve"], color="black", ls=style, lw=1.5, label=n["name"])
    ax.axvline(cfg["analysis"]["early_images"], color="gray", ls=":", lw=1)
    ax.set(xscale="log", xlabel="training images seen (log scale)", ylabel="validation accuracy",
           title="Learning curves (dotted line = end of the 'early' phase)", ylim=(0.5, None))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


def _per_budget_dots(ax, df, column, cfg):
    """One dot per winner, grouped by budget, one color per method, short line = mean."""
    budgets, methods = cfg["run"]["budgets"], cfg["search"]["methods"]
    for k, m in enumerate(methods):
        offset = (k - (len(methods) - 1) / 2) * 0.18
        for i, b in enumerate(budgets):
            values = df[(df["budget"] == b) & (df["method"] == m)][column]
            ax.scatter([i + offset] * len(values), values, color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                       alpha=0.8, label=m if i == 0 else None)
            ax.hlines(values.mean(), i + offset - 0.07, i + offset + 0.07, color=METHOD_COLOR[m], lw=2)
    ax.set_xticks(range(len(budgets)), [f"{b:,}" for b in budgets])
    ax.set_xlabel("search budget (training images per candidate)")


def early_speed(df, cfg, path):
    """H1: early AUC of every winner, per budget and method, vs the baselines."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _per_budget_dots(ax, df, "early_auc", cfg)
    for (_, b), style in zip(df[df["method"] == "Baseline"].iterrows(), BASELINE_STYLE):
        ax.axhline(b["early_auc"], color="black", ls=style, lw=1, label=b["name"])
    ax.set(ylabel=f"early AUC (avg val accuracy, first {cfg['analysis']['early_images']:,} images)",
           title="H1: how fast do the winners learn at the start? (higher = faster)")
    ax.legend(fontsize=8)
    _save(fig, path)


def speed_vs_accuracy(df, cfg, path):
    """H2: early speed vs final test accuracy. Top right = fast AND good."""
    colors = budget_colors(cfg["run"]["budgets"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, w in df[df["method"] != "Baseline"].iterrows():
        ax.scatter(w["early_auc"], w["test_acc"], color=colors[w["budget"]], marker=METHOD_MARKER[w["method"]], s=50)
    for _, b in df[df["method"] == "Baseline"].iterrows():
        ax.scatter(b["early_auc"], b["test_acc"], color="black", marker="*", s=250)
        ax.annotate(b["name"].replace("Baseline ", ""), (b["early_auc"], b["test_acc"]), fontsize=8,
                    xytext=(6, -3), textcoords="offset points")
    for b, c in colors.items():                                  # legend: colors = budgets, shapes = methods
        ax.scatter([], [], color=c, label=f"budget {b:,}")
    for m, marker in METHOD_MARKER.items():
        ax.scatter([], [], color="gray", marker=marker, label=m)
    ax.set(xlabel="early AUC (speed at the start)", ylabel="final test accuracy",
           title="H2: do fast learners pay with final accuracy? (top right = fast and good)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, path)


def search_progress(runs, cfg, path):
    """Best fitness found so far, per method, at the shortest budget (mean over seeds)."""
    budget, cpg = cfg["run"]["budgets"][0], cfg["search"]["cpg"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in cfg["search"]["methods"]:
        best = np.array([r["best_so_far"] for r in runs if r["budget"] == budget and r["method"] == m])
        networks_trained = np.arange(1, best.shape[1] + 1) * cpg
        ax.plot(networks_trained, best.mean(0), color=METHOD_COLOR[m], marker=METHOD_MARKER[m], label=m)
    ax.set(xlabel="networks trained", ylabel=f"best fitness so far (val accuracy after {budget:,} images)",
           title=f"Search progress: is evolution better than random? (budget {budget:,})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def diversity(div, cfg, path):
    """H1b: QD-score per budget and method."""
    budgets, methods = cfg["run"]["budgets"], cfg["search"]["methods"]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, m in enumerate(methods):
        stats = [div[(div["budget"] == b) & (div["method"] == m)]["qd_score"] for b in budgets]
        ax.bar(np.arange(len(budgets)) + (k - (len(methods) - 1) / 2) * width, [s.mean() for s in stats], width,
               yerr=[s.std(ddof=0) for s in stats], color=METHOD_COLOR[m], label=m, capsize=3)
    ax.set_xticks(range(len(budgets)), [f"{b:,}" for b in budgets])
    ax.set(xlabel="search budget (training images per candidate)", ylabel="QD-score (higher = more diverse good networks)",
           title="H1b: diversity of the networks each method found")
    ax.legend()
    _save(fig, path)
