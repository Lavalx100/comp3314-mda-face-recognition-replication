#!/usr/bin/env bash
#SBATCH --job-name=comp3314-mda
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/mda-orl-%j.out
#SBATCH --error=hpc/logs/mda-orl-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${MDA_PYTHON:-python3}"
ORL_ARCHIVE="${MDA_ARCHIVE:-${HOME}/data/att_faces.zip}"
OUTPUT_DIR="${MDA_OUTPUT_DIR:-results/hpc-full-grid-unit-l2-${SLURM_JOB_ID:-manual}}"

if [[ ! -f "${ORL_ARCHIVE}" ]]; then
  echo "ORL archive not found: ${ORL_ARCHIVE}" >&2
  echo "Set MDA_ARCHIVE to the uploaded Cambridge att_faces.zip path." >&2
  exit 2
fi

# One BLAS thread per fit; the Python sweep parallelizes independent fits.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

resume_args=()
if [[ "${MDA_RESUME:-0}" == "1" ]]; then
  resume_args=(--resume)
fi

"${PYTHON_BIN}" scripts/run_orl_dimension_sweep.py \
  --archive "${ORL_ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --full-mda-grid \
  --train-per-class 2,3,4,5 \
  --repeats 5 \
  --seed 3314 \
  --histogram-levels 64 \
  --projection-normalization unit_l2 \
  --mda-stopping paper_figure \
  --mda-order row_first \
  --paper-epsilon 1e-4 \
  --max-iter 10 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "${resume_args[@]}"
