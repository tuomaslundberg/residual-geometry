#!/bin/bash
#SBATCH --job-name=reg-geo-baseline
#SBATCH --account=project_462000999
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/slurm/logs/%x_%j.out
#SBATCH --error=experiments/slurm/logs/%x_%j.err

# LUMI: load modules and activate environment
module purge
module load LUMI/24.03 partition/G
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240315

# Activate venv (adjust path)
source /scratch/project_462000999/tlundber/venvs/register-geometry/bin/activate

# Repo root
REPO=/scratch/project_462000999/tlundber/register-geometry

cd $REPO

python scripts/run_pipeline.py \
    --config experiments/configs/exp_baseline.yaml

echo "Done: $SLURM_JOB_ID"
