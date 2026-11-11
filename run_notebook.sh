#!/bin/bash
#SBATCH --job-name=seminar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --account=ealloc_ati-1-ltat-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out

# Runs the whole seminar notebook on a GPU node, no browser needed.
#   sbatch run_notebook.sh          -> full run
#   sbatch run_notebook.sh test     -> quick check (a few minutes)
# Output: notebooks/results_<mode>_<date>.json + notebooks/executed_<mode>.ipynb (with all plots)

module load python/3.12.13
source .venv/bin/activate
cd notebooks
export RUN_MODE=${1:-full}
jupyter nbconvert --to notebook --execute Seminar_v3.ipynb \
    --output "executed_${RUN_MODE}.ipynb" --ExecutePreprocessor.timeout=-1
