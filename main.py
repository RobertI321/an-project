"""Run the whole experiment:   python main.py              (uses config.yaml)
                              python main.py config_test.yaml
Results go to results/<date-time>/: report.txt, figures.png, results.pkl, config.yaml
"""
import pickle
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml

from src.analysis import add_diversity, add_metrics, plot, report
from src.data import load_data
from src.genome import decode, describe
from src.search import SEARCH_CLASSES, run_search
from src.train import fitness, train_population


# Load in config, create output folder, copy config.yaml there for reproducibility
config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
cfg = yaml.safe_load(open(config_path))
out = Path("results") / time.strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True)
shutil.copy(config_path, out / "config.yaml")             # keep the exact settings with the results
print("JAX backend:", jax.default_backend())
data = load_data(cfg["dataset"])

# 1. Search: every (budget, method, seed) finds one winner
runs, t0 = [], time.time()
for budget in cfg["budgets"]:
    for name in cfg["methods"]:
        for seed in cfg["seeds"]:
            t = time.time()
            fit_fn = lambda hps: fitness(hps, data, budget, seed)
            G, F, best = run_search(SEARCH_CLASSES[name](seed), fit_fn, cfg["cpg"], cfg["generations"])

            # re-check the top_k best with new seeds; the best average is the winner
            top_hps = [decode(g) for g in G[np.argsort(F)[::-1][:cfg["top_k"]]]]
            recheck = np.mean([fitness(top_hps, data, budget, seed=1000 + 10 * seed + r)
                               for r in range(cfg["recheck_seeds"])], axis=0)
            hp = top_hps[np.argmax(recheck)]
            runs.append(dict(budget=budget, method=name, seed=seed, G=G, F=F, best=best, hp=hp))
            print(f"budget={budget:>7} {name:<14} seed {seed}: best {F.max():.4f} "
                  f"rechecked {recheck.max():.4f} [{time.time() - t:.0f}s]  {describe(hp)}")
print(f"All searches done in {time.time() - t0:.0f}s")

# 2. Retrain the baselines + all winners from scratch, several seeds each
configs = [dict(budget=None, method="Baseline", name=k, hp=hp) for k, hp in cfg["baselines"].items()] + runs
hps = [c["hp"] for c in configs]
n_images = cfg["epochs"] * len(data[0])
print(f"Retraining {len(hps)} networks x {cfg['final_seeds']} seeds for {n_images} images...")
final = [train_population(hps, data, n_images, seed=777 + r, eval_every=cfg["eval_every"])
         for r in range(cfg["final_seeds"])]
curves = np.stack([f[1] for f in final], axis=1)           # (networks, seeds, eval points)
test = np.stack([f[2] for f in final], axis=1)             # (networks, seeds)

# 3. Analyse and save
add_metrics(configs, curves, test, cfg)
add_diversity(runs, cfg)
text = report(configs, cfg)
print(text)
(out / "report.txt").write_text(text, encoding="utf-8")
plot(configs, cfg).savefig(out / "figures.png", dpi=110)
pickle.dump(configs, open(out / "results.pkl", "wb"))      # everything, for later analysis in a notebook
print(f"Saved to {out}  [{time.time() - t0:.0f}s total]")
