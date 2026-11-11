"""Everything that gets written into the results folder:

results/<date>_<mode>/
    summary.md      what was tested, how, and the results in plain language  <- read this first
    config.yaml     the exact settings of this run
    results.json    all raw data (same format as the team notebook -> load it there with LOAD_RESULTS)
    results.pkl     the same data for Python (numpy arrays stay arrays)
    networks.csv    one row per retrained network: hyperparameters, speed, accuracy
    diversity.csv   one row per search: grid coverage, QD-score, good cells
    tests.csv       all statistical tests
    figures/*.png
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from src import plots
from src.analysis import diversity_table, hypothesis_tests, network_table, summary_table


def save_results(cfg, runs, networks, run_info):
    """Save the raw data (config, json, pkl) and return the results folder."""
    out = Path("results") / f"{time.strftime('%Y-%m-%d_%H%M')}_{cfg['run_mode']}"
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in cfg.items() if k != "run"}
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run = cfg["run"]
    settings = dict(RUN_MODE=cfg["run_mode"], BUDGETS=run["budgets"], SEEDS=run["seeds"], GENERATIONS=run["generations"],
                    EPOCHS=run["epochs"], TOP_K=run["top_k"], RECHECK_SEEDS=run["recheck_seeds"],
                    SEARCH_METHODS=cfg["search"]["methods"], CPG=cfg["search"]["cpg"],
                    FINAL_SEEDS=cfg["retrain"]["seeds"], EVAL_EVERY=cfg["retrain"]["eval_every"],
                    THRESHOLD=cfg["analysis"]["thresholds"], SEARCH_SPACE=cfg["search_space"],
                    BASELINES=cfg["baselines"], run_info=run_info)
    data = dict(runs=runs, networks=networks, settings=settings)

    def to_plain(obj):                          # numpy -> plain Python for JSON
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))

    with open(out / "results.json", "w") as f:
        json.dump(data, f, default=to_plain)
    with open(out / "results.pkl", "wb") as f:
        pickle.dump(data, f)
    return out


def write_report(out, cfg, space, runs, networks, run_info):
    """Tables, tests, figures and summary.md."""
    df = network_table(networks, cfg)
    div = diversity_table(runs, space, cfg)
    summary = summary_table(df, div)
    tests = hypothesis_tests(df, div, cfg)
    df.to_csv(out / "networks.csv", index=False)
    div.to_csv(out / "diversity.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plots.learning_curves(networks, cfg, fig_dir / "1_learning_curves.png")
    plots.early_speed(df, cfg, fig_dir / "2_early_speed.png")
    plots.speed_vs_accuracy(df, cfg, fig_dir / "3_speed_vs_accuracy.png")
    plots.search_progress(runs, cfg, fig_dir / "4_search_progress.png")
    plots.diversity(div, cfg, fig_dir / "5_diversity.png")

    text = summary_markdown(cfg, df, summary, tests, run_info, len(runs), len(networks))
    (out / "summary.md").write_text(text, encoding="utf-8")


def _verdicts(tests, cfg):
    """One plain-language line per hypothesis."""
    def row(hypothesis, comparison):
        r = tests[(tests["hypothesis"] == hypothesis) & (tests["comparison"] == comparison)]
        return None if r.empty else r.iloc[0]

    def verdict(r):
        return "**supported**" if r["significant"] else "**not shown**"

    budgets = cfg["run"]["budgets"]
    lines = []
    r = row("H1 faster than best baseline", "all budgets")
    lines.append(f"- **H1, faster than the baselines:** {verdict(r)}. {r['result']} at the start than the best "
                 f"baseline (p = {r['p']:.4f}).")
    r = row("H1 short budget faster early", "all methods: " + f"{budgets[0]} > {budgets[-1]}")
    if r is not None:
        lines.append(f"- **H1, short budget gives faster learners:** {verdict(r)}. Early AUC of {budgets[0]:,}-image "
                     f"winners vs {budgets[-1]:,}-image winners: {r['result']} (p = {r['p']:.4f}).")
    r = row("H2 long budget better at the end", "all methods: " + f"{budgets[-1]} > {budgets[0]}")
    if r is not None:
        lines.append(f"- **H2, fast learners end lower:** {verdict(r)}. Final test accuracy of {budgets[-1]:,}-image "
                     f"winners vs {budgets[0]:,}-image winners: {r['result']} (p = {r['p']:.4f}).")
    for hypothesis, label in [("Method beats random search", "Search methods beat random search"),
                              ("H1b MAP-Elites more diverse", "H1b, MAP-Elites is more diverse")]:
        rows = tests[tests["hypothesis"] == hypothesis]
        if len(rows):
            lines.append(f"- **{label}:** significant in {int(rows['significant'].sum())} of {len(rows)} comparisons "
                         f"(see the test table).")
    return lines


def _tidy(table):
    """Round for reading: speeds to whole images, accuracies to 4 decimals."""
    speed = [c for c in table.columns if c.startswith("speed_to")]
    return table.round(4).round({c: 0 for c in speed})


def summary_markdown(cfg, df, summary, tests, run_info, n_searches, n_networks):
    run, search, retrain, a = cfg["run"], cfg["search"], cfg["retrain"], cfg["analysis"]
    per_search = search["cpg"] * run["generations"]
    space_rows = "\n".join(
        f"| {name} | {g['type']} | " + (", ".join(map(str, g["options"])) if g["type"] == "choice" else f"{g['min']} .. {g['max']}") + " |"
        for name, g in cfg["search_space"].items())
    baselines = df[df["method"] == "Baseline"].set_index("name")[[c for c in df.columns if c.startswith("speed_to")]
                                                                  + ["acc_at", "early_auc", "final_val", "test_acc"]]
    warning = ("\n> **Note:** only {n} seed(s) per method. With so few data points the per-method tests can't reach "
               "p < 0.05; use `--mode full` (5 seeds) for real conclusions.\n".format(n=len(run["seeds"]))
               if len(run["seeds"]) < 5 else "")

    return f"""# Fast learners: results ({cfg['run_mode']} run, {time.strftime('%Y-%m-%d %H:%M')})

## What is tested
If networks are selected **only by their validation accuracy after a short training budget**,
does the search end up finding networks that learn fast?

- **H1:** winners of short-budget searches learn faster at the start than hand-picked baselines,
  and faster than winners of long-budget searches.
- **H2:** ...but they end with a lower final accuracy (sprinters vs stayers).
- **H1b:** MAP-Elites finds a more diverse set of good networks than the other methods.

## How
1. **Search.** For every budget x method x seed, one search runs {run['generations']} generations of {search['cpg']} networks
   = {per_search} networks. Each network is trained from scratch on only `budget` images;
   its **fitness** = validation accuracy after that.
   - budgets (images per candidate): {', '.join(f'{b:,}' for b in run['budgets'])}
   - methods: {', '.join(search['methods'])}
   - seeds: {run['seeds']}  ->  {n_searches} searches
2. **Pick a winner.** The top {run['top_k']} of each search are trained again on {run['recheck_seeds']} new seed(s);
   the best average wins. -> {n_searches} winners
3. **Retrain.** All winners + {len(cfg['baselines'])} baselines are trained from scratch on {run['epochs']} epochs
   ({run['epochs'] * 50_000:,} images), {retrain['seeds']} times each (seeds {retrain['first_seed']}+).
   Validation accuracy is measured every {retrain['eval_every']} images -> learning curve.
4. **Measure.**
   - speed_to_XX% = images needed to reach XX% validation accuracy (fewer = faster)
   - acc_at = validation accuracy after {a['acc_at']:,} images
   - early_auc = average validation accuracy over the first {a['early_images']:,} images (higher = faster start)
   - test_acc = test accuracy at the end
   - diversity: all networks of a search go into a {search['grid_bins']}x{search['grid_bins']} grid
     (learning rate x network size); QD-score = sum of (best accuracy - 0.1) over the used cells

**Search space**

| gene | type | range |
|---|---|---|
{space_rows}

## Results
{warning}
### Verdicts (p < 0.05 = supported)
{chr(10).join(_verdicts(tests, cfg))}

### Winners: averages per budget and method
```
{_tidy(summary).to_string()}
```

### Baselines
```
{_tidy(baselines).to_string()}
```

### All statistical tests
```
{tests.round(4).to_string(index=False)}
```

## Files in this folder
- `figures/` - 1 learning curves, 2 early speed (H1), 3 speed vs final accuracy (H2), 4 search progress, 5 diversity (H1b)
- `networks.csv` - every retrained network (hyperparameters + all metrics), opens in Excel
- `diversity.csv`, `tests.csv` - per-search diversity and all tests
- `results.json` / `results.pkl` - all raw data (load the json in the team notebook with `LOAD_RESULTS`)
- `config.yaml` - exact settings of this run

## Run info
- device: {run_info['device']}
- runtime: {run_info['runtime_min']:.1f} min
- {n_searches} searches x ~{per_search + run['top_k'] * run['recheck_seeds']} trainings + {n_networks} x {retrain['seeds']} retrainings
"""
