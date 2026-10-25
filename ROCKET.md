# Running on Rocket (UT HPC)

## Connect to Rocket

1. Make an SSH key (on your laptop, once):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_rocket
   ```
2. Upload the **public** key (`~/.ssh/id_ed25519_rocket.pub`) to your ETAIS profile. It can take up to an hour to reach Rocket.
3. Add to `~/.ssh/config` (Windows: `C:\Users\<you>\.ssh\config`):
   ```
   Host rocket
       HostName rocket.hpc.ut.ee
       User hpc_<your_username>
       IdentityFile ~/.ssh/id_ed25519_rocket
   ```
4. Test in a terminal: `ssh rocket`
5. VS Code: install the **Remote - SSH** extension → `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `rocket`. Then File → Open Folder → `~/an_project`.

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
sbatch job.sh config_test.yaml   # quick test, ~1 min
sbatch job.sh                    # full run with config.yaml
squeue -u $USER                  # PD = waiting, R = running
tail -f slurm-<jobid>.out        # watch output
```
Results: `results/<date-time>/` (report.txt, figures.png).

## Run a notebook on a GPU

1. Start Jupyter on a GPU node:
   ```bash
   sbatch jupyter.sh
   squeue -u $USER                               # wait until R
   grep -E "NODE|http://" jupyter-<jobid>.log    # copy the URL with the node name
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
- Don't run JAX on the login node, it has no GPU.
- `PD (Resources)` / `PD (Priority)` = waiting for a GPU, starts by itself.
- `job.sh` / `jupyter.sh` use the project account `ealloc_ati-1-ltat-9` and `--gres=gpu:tesla:1`.

## Slurm cheat sheet

```bash
sbatch job.sh                     # submit a job
sbatch job.sh config_test.yaml    # submit with an argument (here: which config)
squeue -u $USER                   # my jobs (PD = waiting, R = running)
squeue -j <jobid> --start         # estimated start time
scancel <jobid>                   # stop one job
scancel -u $USER                  # stop ALL my jobs
scancel -n jupyter-an             # stop jobs by name
tail -f slurm-<jobid>.out         # follow output live (Ctrl+C stops watching, not the job)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # finished job: status, time, memory used
sinfo -s                          # partitions and free nodes
sinfo -p gpu -o "%N %G"           # GPU types per node
sacctmgr show assoc user=$USER format=account%30       # my accounts
srun -p gpu --gres=gpu:tesla:1 -A ealloc_ati-1-ltat-9 --time=00:30:00 --pty bash   # interactive shell on a GPU node
```