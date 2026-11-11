# Fast learners (ANI Project 17)

If networks are selected only by their accuracy after a **short** training budget, does the search find
networks that learn fast? Compares GA, MAP-Elites, Optuna TPE and random search.

```
config.yaml          ALL settings: run size (test / quick / full), search space, baselines, metrics
main.py              runs everything: search -> pick winners -> retrain -> save + report
src/
  config.py          reads config.yaml
  data.py            Fashion-MNIST: download once (data/), scale to [0,1], 50k/10k split
  search_space.py    genome (numbers in [0,1]) <-> hyperparameters, diversity grid
  train.py           the MLP + training many networks in parallel on the GPU
  search.py          GA, MAP-Elites, Optuna TPE, random search
  analysis.py        tables (pandas) + statistical tests
  plots.py           figures
  report.py          writes the results folder + summary.md
notebooks/           the team's Colab notebook (Seminar_v3 can load our results.json)
job.sh               Rocket: run main.py on a GPU          (see ROCKET.md)
run_notebook.sh      Rocket: run the notebook on a GPU
```

## Run

```bash
pip install -r requirements.txt
python main.py --mode test       # a few minutes, checks everything works
python main.py --mode full       # the real experiment (GPU, ~3 h)
```
On Rocket: `sbatch job.sh test` / `sbatch job.sh full` (see [ROCKET.md](ROCKET.md)).

## Results

Each run creates `results/<date>_<mode>/`:

| file | what |
|---|---|
| `summary.md` | **read this first**: what was tested, how, results + verdicts in plain language |
| `figures/` | learning curves, early speed (H1), speed vs final accuracy (H2), search progress, diversity (H1b) |
| `networks.csv` | every retrained network: hyperparameters, speed, accuracy (opens in Excel) |
| `diversity.csv`, `tests.csv` | diversity per search, all statistical tests |
| `results.json` | all raw data, same format as the team notebook -> set `LOAD_RESULTS` there to analyse in Colab |
| `results.pkl` | same data for Python |
| `config.yaml` | exact settings of the run |

## Notes
- Budgets are counted in **training images**, not steps, so batch size doesn't change how much data a network gets.
- Every network sees the same images in the same order; only the starting weights differ (seeds).
- Same seeds as the team notebook (retraining seeds 500+), so both give the same numbers.
