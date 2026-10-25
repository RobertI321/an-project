# Fast learners (ANI Project 17)

Does selecting networks by their accuracy after a **short** training budget find fast learners?
Compares GA, MAP-Elites, Optuna TPE and random search.

```
config.yaml        all settings (quick run; full-run values in comments)
config_test.yaml   tiny test run, like TEST_RUN in the notebook
main.py            runs everything: search -> retrain winners -> report
job.sh             Slurm job for Rocket: runs main.py on a GPU
jupyter.sh         Slurm job for Rocket: Jupyter on a GPU node
ROCKET.md          how to connect to Rocket, run jobs + notebooks, Slurm cheat sheet
src/
  data.py          download (cached in data/), scale to [0,1], 50k/10k split
  genome.py        search space: 6 genes in [0,1] -> hyperparameters
  train.py         MLP + parallel training on the GPU (jax.vmap)
  search.py        GA, MAP-Elites, Optuna TPE, random search (propose / update)
  analysis.py      metrics, statistical tests, plots
notebooks/         the team's Colab notebook
```

## Run

**On Rocket (recommended):** see [ROCKET.md](ROCKET.md).

**Locally / anywhere else:**
```bash
pip install -r requirements.txt
python main.py                    # uses config.yaml
python main.py config_test.yaml   # quick check that everything works
```
Real runs need a GPU (plain `jax` is CPU-only; on a Linux GPU machine also run `pip install -U "jax[cuda12]"`).

Results land in `results/<date-time>/`: `report.txt`, `figures.png`, `results.pkl` and a copy of the config.

## Notes
- Training budgets are counted in **images seen**, not steps, so batch size doesn't change how much data a network gets.
- Every network sees the same images in the same order (`DATA_SEED` in `src/train.py`); only the starting weights differ.
- The data is downloaded once into `data/` (~30 MB). `data/`, `results/` and `.venv/` are not in git.