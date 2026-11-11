"""Fast learners: if networks are selected only by their accuracy after a SHORT training budget,
does the search find networks that learn fast?

    python main.py                 # run_mode from config.yaml
    python main.py --mode full     # test | quick | full

Steps: 1) search  2) pick one winner per search  3) retrain winners + baselines  4) save + report.
Everything ends up in results/<date>_<mode>/ - start with summary.md there.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window needed (Rocket / Colab)
import jax                                # noqa: E402
import numpy as np                        # noqa: E402

from src.config import load_config                        # noqa: E402
from src.data import load_data                            # noqa: E402
from src.report import save_results, write_report        # noqa: E402
from src.search import SEARCH_METHODS, run_search         # noqa: E402
from src.search_space import SearchSpace, describe        # noqa: E402
from src.train import Trainer                             # noqa: E402


def search_all(cfg, space, trainer):
    """Steps 1 + 2: every (budget, method, seed) search returns one winner."""
    run, runs = cfg["run"], []
    for budget in run["budgets"]:
        for method in cfg["search"]["methods"]:
            for seed in run["seeds"]:
                start = time.time()
                fitness_fn = lambda hps: trainer.fitness(hps, budget, seed)
                genomes, fitness, best_so_far = run_search(
                    SEARCH_METHODS[method](seed, space), space, fitness_fn, cfg["search"]["cpg"], run["generations"])

                # re-check the top_k on fresh seeds; best average wins (filters out lucky networks)
                top = genomes[np.argsort(fitness)[::-1][:run["top_k"]]]
                top_hps = [space.decode(g) for g in top]
                recheck = np.mean([trainer.fitness(top_hps, budget, seed=1000 + 10 * seed + r)
                                   for r in range(run["recheck_seeds"])], axis=0)
                winner = top[np.argmax(recheck)]

                runs.append(dict(budget=budget, method=method, seed=seed, genomes=genomes, fitness=fitness,
                                 best_so_far=best_so_far, winner=winner, winner_recheck=float(recheck.max())))
                print(f"budget={budget:>7} {method:<14} seed {seed}: best {fitness.max():.4f} "
                      f"rechecked {recheck.max():.4f} [{time.time() - start:.0f}s]  {describe(space.decode(winner))}")
    return runs


def retrain_all(cfg, space, trainer, runs):
    """Step 3: retrain the baselines + all winners from scratch, longer and with several seeds."""
    networks = [dict(name=f"Baseline {name}", method="Baseline", budget=None, seed=None, hp=hp)
                for name, hp in cfg["baselines"].items()]
    networks += [dict(name=f"{r['method']} B={r['budget']} s{r['seed']}", method=r["method"],
                      budget=r["budget"], seed=r["seed"], hp=space.decode(r["winner"])) for r in runs]

    hps = [net["hp"] for net in networks]
    n_images = cfg["run"]["epochs"] * trainer.n_train
    retrain = cfg["retrain"]
    print(f"Retraining {len(hps)} networks x {retrain['seeds']} seeds on {n_images:,} images...")
    results = [trainer.train(hps, n_images, seed=retrain["first_seed"] + s, eval_every=retrain["eval_every"])
               for s in range(retrain["seeds"])]

    for i, net in enumerate(networks):
        net["seed_curves"] = np.stack([curves[i] for curves, _ in results])   # (seeds, eval points)
        net["curve"] = net["seed_curves"].mean(axis=0)
        net["test_accuracy"] = float(np.mean([test[i] for _, test in results]))
    return networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", help="test | quick | full (overrides run_mode in the config)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    print(f"Run mode: {cfg['run_mode']} | JAX backend: {jax.default_backend()} ({jax.devices()[0]})")
    space = SearchSpace(cfg["search_space"], cfg["search"]["grid_bins"])
    data = load_data(cfg["dataset"])
    trainer = Trainer(data, space.max_width)

    start = time.time()
    runs = search_all(cfg, space, trainer)
    networks = retrain_all(cfg, space, trainer, runs)
    run_info = dict(device=str(jax.devices()[0]), runtime_min=(time.time() - start) / 60)

    out = save_results(cfg, runs, networks, run_info)           # step 4
    write_report(out, cfg, space, runs, networks, run_info)
    print(f"\nDone in {run_info['runtime_min']:.0f} min. Results: {out}/summary.md")


if __name__ == "__main__":
    main()
