# HKU HPC2021 CPU run

This is an optional, ready-to-submit SLURM configuration for the full 10,130-pair MDA/2-2 grid. It is set up for one AMD CPU node with 32 fit workers and single-threaded BLAS. It does not submit a job or assume that an account is available.

## Access and setup

HKU Research Computing requires authorization. Its current [account guidance](https://hpc.hku.hk/documentation/access/) says research projects must be led by a full-time professoriate PI, and undergraduate access is limited, usually through a PI's group. The account request requires PI endorsement and a project presentation of at least five slides describing the work, progress, and need for HPC. Check eligibility with a faculty sponsor before planning around this cluster. See the current [HPC2021 partition and QoS limits](https://hpc.hku.hk/guide/slurm-guide/slurm-partitions-qos/) and [SLURM job-script examples](https://hpc.hku.hk/guide/slurm-guide/slurm-job-script/).

Once an account is approved:

1. Connect from the HKU network or HKUVPN and SSH to `hpc2021.hku.hk`.
2. Copy this repository and the Cambridge ORL archive to your account. Do not put the face archive in Git.
3. On the login node, inspect available Python modules with `module avail python`, load an available Python 3.12 module, and create an environment with the repository's `requirements.txt`. The site module name can change, so the script accepts the interpreter path through `MDA_PYTHON`.
4. Upload the ORL zip to a private data directory, for example `$HOME/data/att_faces.zip`.
5. From the repository root, submit the job:

   ```bash
   sbatch hpc/orl_mda_grid.sbatch
   ```

The job writes its Slurm logs under `hpc/logs/` and defaults to a job-specific output folder such as `results/hpc-full-grid-unit-l2-<job-id>/`. It checkpoints each completed gallery/repeat split. To resume an interrupted run, preserve the original output directory and pass that same path explicitly when resubmitting:

```bash
MDA_RESUME=1 MDA_OUTPUT_DIR="results/hpc-full-grid-unit-l2-<original-job-id>" \
  sbatch hpc/orl_mda_grid.sbatch
```

Optional environment overrides:

```bash
MDA_PYTHON="$HOME/venvs/mda/bin/python" \
MDA_ARCHIVE="$HOME/data/att_faces.zip" \
MDA_OUTPUT_DIR="results/hpc-full-grid-unit-l2" \
sbatch hpc/orl_mda_grid.sbatch
```

The script requests the current general AMD partition (`amd`) and normal QoS, one node, 32 CPUs, 16 GB RAM, and four hours. The latest corrected local M3 Max benchmark measured 47.48 fits/s at 14 workers; projecting the full grid from that representative batch gives about 71 minutes. This is not a full-run measurement. A 32-core AMD allocation's per-core speed, queue delay, account eligibility, and package setup have not been measured. Run a short pilot after access is granted before moving any full workload.

## Why CPU instead of GPU

The experiment repeatedly solves small 112×112 and 92×92 dense eigenproblems and updates one independent dimension pair at a time. This Mac has no CUDA runtime and neither PyTorch nor MLX is installed. A GPU version would require a separate solver port and numerical validation. Current evidence shows the CPU implementation is already practical after caching; no GPU port or remote run is included here.
