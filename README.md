# Fast learners (ANI Project 17)

Does selecting networks by their accuracy after a **short** training budget find fast learners?
Compares GA, MAP-Elites, Optuna TPE and random search.

```
config.yaml        all settings (quick run; full-run values in comments)
config_test.yaml   tiny test run, like TEST_RUN in the notebook
main.py            runs everything: search -> retrain winners -> report
src/
  data.py          download (cached in data/), scale to [0,1], 50k/10k split
  genome.py        search space: 6 genes in [0,1] -> hyperparameters
  train.py         MLP + parallel training on the GPU (jax.vmap)
  search.py        GA, MAP-Elites, Optuna TPE, random search (propose / update)
  analysis.py      metrics, statistical tests, plots
notebooks/         the team's Colab notebook
```

## Run

```bash
pip install -r requirements.txt
python main.py                 # uses config.yaml
python main.py config_test.yaml  # quick check that everything works
```

Results land in `results/<date-time>/`: `report.txt`, `figures.png`, `results.pkl` and a copy of the config.
Needs a GPU for real runs (Colab: `!git clone ...`, `%cd an-project`, `!pip install -r requirements.txt`, `!python main.py`).

Training budgets are counted in **samples seen**, not steps, so batch size doesn't change how much data a network gets.
