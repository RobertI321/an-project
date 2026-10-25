#!/bin/bash
#SBATCH --job-name=jupyter-an
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --account=ealloc_ati-1-ltat-9
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=jupyter-%j.log

module load python/3.12.13
source ~/an_project/.venv/bin/activate
PORT=$(shuf -i 20000-29999 -n 1)
echo "==> NODE=$(hostname -s) PORT=$PORT"
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --notebook-dir=$HOME/an_project