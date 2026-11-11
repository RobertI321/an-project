# Running on Rocket (UT HPC)

## One-time setup

```bash
git clone https://github.com/RobertI321/an-project.git ~/an_project
cd ~/an_project
module load python/3.12.13
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt jupyterlab flax "jax[cuda12]"
JAX_PLATFORMS=cpu python -c "from src.data import load_data; load_data('fashion_mnist')"   # download data once
```

## Run the experiment (no notebook)

```bash
sbatch job.sh test               # quick check, a few minutes
sbatch job.sh                    # full run (~3 h)
squeue -u $USER                  # PD = waiting, R = running
tail -f slurm-<jobid>.out        # watch output
```
Results: `results/<date>_<mode>/` - start with `summary.md`.

## Data for the notebook (once, on the login node)

GPU nodes may have no internet, so download everything the notebook needs first:
```bash
cd ~/an_project
mkdir -p data/fmnist-c
JAX_PLATFORMS=cpu python -c "from src.data import load_data; load_data('fashion_mnist')"     # -> data/fashion_mnist.npz
URL=https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/v1.0.0/generated/npy
curl -sSL -o data/fmnist-c/fmnist-c-test.npy        $URL/fmnist-c-test.npy           # corrupted test images
curl -sSL -o data/fmnist-c/fmnist-c-test-labels.npy $URL/fmnist-c-test-labels.npy
```
The notebook (`notebooks/Seminar_v4.ipynb`) finds these files automatically; in Colab it downloads them itself.

## Run a notebook on a GPU

1. Start Jupyter on a GPU node:
   ```bash
   sbatch jupyter.sh
   squeue -u $USER                                   # wait until R
   grep -E "NODE|http://" jupyter-<jobid>.log        # copy the URL with the node name, e.g. http://falcon4:24567/lab?token=...
   ```
2. In VS Code (connected to Rocket): open the notebook → **Select Kernel → Select Another Kernel → Existing Jupyter Server** → paste the URL → pick **Python 3**.
3. Run the cells. The setup cell should print `JAX backend: gpu`.
4. When done: `scancel <jobid>` (otherwise it holds the GPU for 4 h).

To use the project code in a notebook in `notebooks/`:
```python
import sys; sys.path.append("..")
from src.data import load_data
x_train, y_train, x_val, y_val, x_test, y_test = load_data("fashion_mnist")
```

## Notes
- Don't run JAX directly on the login node: it has no GPU.
- `PD (Resources)` / `PD (Priority)` = waiting for a GPU, it starts by itself.
- Jobs use the project account `ealloc_ati-1-ltat-9` and `--gres=gpu:tesla:1` (set in `job.sh` / `jupyter.sh`).
