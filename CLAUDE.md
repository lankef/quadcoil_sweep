# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a **QUADCOIL coil optimization preprocessing pipeline** for stellarator fusion reactor design. It downloads plasma equilibrium configurations from the [Constellaration dataset](https://huggingface.co/datasets/proxima-fusion/constellaration) on Hugging Face, filters them by stability criteria, and saves processed inputs (`quadcoil_inputs.npy`) for downstream QUADCOIL coil optimization runs.

## Running the Script

```bash
# Activate the conda environment first
conda activate desc

# Run locally (processes 5 configs by default)
python download_constellaration.py --device_scaling stellaris --nfp 3

# Run with more configs
python download_constellaration.py --device_scaling stellaris --num_config 100 --nfp 3

# Submit to SLURM
sbatch jobscript.sh
```

**CLI arguments:**
- `--device_scaling`: Device type from `config.json` — one of `stellaris`, `aries`, `w7x`, `squid_040925` (default: `stellaris`)
- `--num_config`: Max number of equilibria to process (default: 5)
- `--nfp`: Number of field periods to filter for (required for meaningful filtering)

## Architecture

### Data Pipeline

```
proxima-fusion/constellaration (Hugging Face)
  "default" config (metadata)  →  filter by nfp + vacuum_well >= 0
  "vmecpp_wout" config         →  filter by plasma_config_id match
      ↓
  Parse VmecppWOut JSON → simsopt equilibrium object
      ↓
  Extract: plasma boundary surface DOFs, net poloidal current
      ↓
  Save: output_nfp=X/<plasma_config_id>/quadcoil_inputs.npy
```

### Key Design Decisions

- **Streaming mode**: Both datasets are loaded with `streaming=True` to avoid downloading the full dataset to disk. A two-pass approach is used: first collect valid `plasma_config_id`s from metadata, then filter the wout dataset by those IDs.
- **SSL workarounds**: HPC clusters often have outdated CA certs. The code globally disables SSL verification via monkey-patching `requests.Session.request`, `httpx.Client.__init__`, and setting env vars. This is intentional for the cluster environment.
- **Cluster detection**: Checks hostname patterns (`vip*`, `rav*`) and `SLURM_JOB_ID` env var. On non-cluster runs, a default `--device_scaling=stellaris` arg is injected.
- **Plasma surface resolution**: All surfaces are forced to `mpol_max_plasma=4`, `ntor_max_plasma=4` (from `config.json`) to ensure uniform JAX compilation across configurations.
- **Code archival**: `save_myself()` copies the entire source directory into `output_nfp=X/src/` for reproducibility.

### Output Structure

```
output_nfp=X/
├── run_config.json              # Serialized RunConfig
├── src/                         # Snapshot of this source directory
└── <plasma_config_id>/
    ├── quadcoil_inputs.npy      # Dict with: nfp, stellsym, plasma_dofs,
    │                            #   net_poloidal_current_amperes, Bnormal_plasma
    └── validation_error.txt     # Present only if VMEC parsing failed
```

### Configuration (`config.json`)

Stores device scaling parameters (minor radius `a`, coil-to-surface distance `d_cs`, coil-to-coil distance `d_cc`, max curvature `k`, field strength `B_0`, coil count `ncoils`) and Fourier resolution settings (`mpol_quadcoil`, `ntor_quadcoil`, `mpol_max_plasma`, `ntor_max_plasma`).

### Key Dependencies

- `datasets` (Hugging Face) — streaming dataset access
- `constellaration.mhd.vmec_utils` — `VmecppWOut` Pydantic model + `as_simsopt_vmec()`
- `simsopt` — plasma surface and coil geometry objects
- `quadcoil` — downstream coil optimization (inputs prepared here)
- `jax.numpy` — used for saving `.npy` files

## SLURM Job

`jobscript.sh` is configured for a single serial job (1 CPU, 10GB RAM, 1 GPU, 30min) on account `torch_pr_292_courant`. Logs go to `../logs/`. The `desc` conda environment must be available on the cluster.
