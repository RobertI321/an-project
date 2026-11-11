#!/bin/bash
#SBATCH --job-name=fastlearners
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --account=ealloc_ati-1-ltat-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out

# sbatch job.sh test   -> a few minutes
# sbatch job.sh quick  -> ~15 min
# sbatch job.sh        -> full run (~3 h)
module load python/3.12.13
source .venv/bin/activate
python -c "import jax; print(jax.devices())"
python -u main.py --mode ${1:-full}
