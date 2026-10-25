#!/bin/bash
#SBATCH --job-name=fastlearners
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --account=ealloc_ati-1-ltat-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

module load python/3.12.13
source .venv/bin/activate
python -c "import jax; print(jax.devices())"
python main.py ${1:-config.yaml}