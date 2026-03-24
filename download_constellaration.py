import shutil
import json
import argparse
import math
from copyreg import pickle
import jax.numpy as jnp
import numpy as np
import datasets
from simsopt import save
from constellaration.mhd import vmec_utils
import os
import ssl
import time
import csv
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# SSL / HTTPS workarounds
# ---------------------------------------------------------------------------
# The cluster (or the dev machine) may have outdated or missing CA certificates,
# causing HTTPS connections to Hugging Face / dataset servers to fail with
# "certificate verify failed". The block below disables all SSL verification
# as a quick workaround — NOT recommended for production/security-sensitive code.

# 1. Tell Python"s built-in ssl module to skip certificate verification globally.
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Suppress the "InsecureRequestWarning" that urllib3 prints every time an
#    unverified HTTPS request is made (since we"re intentionally doing this).
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. Clear the CA bundle env-vars so that curl-based tools also skip verification.
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# ---------------------------------------------------------------------------
# Monkey-patch `requests` to always set verify=False
# ---------------------------------------------------------------------------
# The `datasets` library uses the `requests` library under the hood for HTTP
# calls. Even though we set env-vars above, `requests` checks them only at
# Session creation time, so we patch the Session.request method directly to
# force verify=False on every call regardless of how the session was created.
import requests
import datetime
from os.path import join
from os import makedirs
from pathlib import Path
from pyevtk.hl import polyLinesToVTK

# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------
# Detect whether the script is running on the HPC cluster by checking the
# hostname (cluster head-nodes are named "vip*" or "rav*") or by the presence
# of the SLURM_JOB_ID environment variable (set automatically by the SLURM
# scheduler when a job is submitted).
import socket
RUNNING_ON_CLUSTER = False
if "vip" in socket.gethostname() or "rav" in socket.gethostname() or "SLURM_JOB_ID" in os.environ: 
    # # On the cluster, OpenMPI shared libraries live in a non-standard path.
    # # We append that path to LD_LIBRARY_PATH so the dynamic linker can find them.
    # if "LD_LIBRARY_PATH" in os.environ:
    #     os.environ["LD_LIBRARY_PATH"] += ":" + str(join(os.environ["OPENMPI_HOME"], "lib"))        
    # else:
    #     os.environ["LD_LIBRARY_PATH"] = str(join(os.environ["OPENMPI_HOME"], "lib"))
    RUNNING_ON_CLUSTER = True

from simsopt.geo import curves_to_vtk, CurveCurveDistance, LpCurveCurvature, MeanSquaredCurvature
from simsopt.field import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from quadcoil.io import simsopt_coil_from_qp

# ---------------------------------------------------------------------------
# Device scaling parameters
# ---------------------------------------------------------------------------
# Each entry corresponds to a real (or design-study) fusion device and
# stores the physical scales used to set geometry/field constraints:
#   a      – minor radius of the plasma [m]
#   d_cs   – minimum coil-to-plasma-surface distance, expressed as a
#            multiple of `a` (will be multiplied by the actual minor
#            radius at run time)
#   d_cc   – minimum coil-to-coil distance, similarly normalised
#   k      – maximum curvature of the coil centreline [m^-1], normalised
#            by `a` (i.e. k_max = k / a)
#   B_0    – on-axis magnetic field [T]
#   ncoils – number of unique coils per half-period (where applicable)
_config_path = Path(__file__).parent / "config.json"
with open(_config_path) as _f:
    _config = json.load(_f)

device_scaling_dict = _config["device_scaling_dict"]
# Fourier harmonics # of the sheet current
mpol_quadcoil = _config["mpol_quadcoil"]
ntor_quadcoil = _config["ntor_quadcoil"]
# Force all plasma boundaries to have the same mpol and
# ntor to avoid recompilation. The Constellaration paper
# states that each configuration has at most 4 modes.
mpol_max_plasma = _config["mpol_max_plasma"]
ntor_max_plasma = _config["ntor_max_plasma"]

# ---------------------------------------------------------------------------
# Monkey-patch `requests.Session.request` to disable SSL verification
# ---------------------------------------------------------------------------
# We wrap the original method and force `verify=False` on every call.
# This covers all HTTP traffic routed through `requests`, including the
# Hugging Face `datasets` library.
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# ---------------------------------------------------------------------------
# Monkey-patch `httpx.Client` to disable SSL verification
# ---------------------------------------------------------------------------
# `huggingface_hub` (used internally by `datasets` for auth and metadata)
# uses `httpx` rather than `requests`. We patch its Client.__init__ to
# pass verify=False so it also skips certificate checking.
import httpx
original_httpx_client_init = httpx.Client.__init__
def patched_httpx_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    return original_httpx_client_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_httpx_client_init

def save_myself(destination):
    """Copy the entire source directory into the output folder.
    
    This ensures the exact code that produced a set of results is archived
    next to the results — useful for debugging and reproducibility.
    """
    source = Path(__file__).parent.resolve()
    dest   = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"Saved self to {dest}")
    
# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def run(run_config):
    # ---------------------------------------------------------------------------
    # Output directory setup
    # ---------------------------------------------------------------------------
    # The output path differs depending on whether we are running locally or on
    # the cluster, and (on the cluster) whether this is a SLURM array job.
    project_dir = Path(__file__).parent.resolve()
    SAVE_DIR = f"{project_dir}/../output_nfp=" + str(run_config.nfp)        
    makedirs(SAVE_DIR, exist_ok=True)
    print("Saving to", SAVE_DIR)
    # Dumping run info
    # Persist the run configuration alongside the outputs for reproducibility.
    with open(join(SAVE_DIR, "run_config.json"), "w") as f:
        json.dump(run_config.__dict__, f, indent=4)


    save_myself(join(SAVE_DIR, "src"))
    # Stream the dataset from Hugging Face instead of downloading it all at once.
    # `streaming=True` returns an IterableDataset, so records are fetched
    # on demand — important when the dataset is large and disk space is limited.
    # `force_redownload` skips any cached version, ensuring we always get fresh data.
    ds = datasets.load_dataset(
        "proxima-fusion/constellaration",
        "vmecpp_wout",        
        split="train",
        streaming=True,
        download_mode="force_redownload"
    )
    # Contains general parameters of the equilibrium. 
    # used for filtering out unstable equilibria and 
    # filter for field periods.
    ds_meta = datasets.load_dataset(
        "proxima-fusion/constellaration",
        "default",
        split="train",
        streaming=True,
        download_mode="force_redownload"
    )
    # Selection criteria for equilibrium.
    # Currently we are interested 
    def criteria_meta(ex_meta):
        nfp_rule = ex_meta["boundary.n_field_periods"] == run_config.nfp
        if not ex_meta["misc.vmecpp_wout_id"]:
            return False
        if ex_meta["metrics.vacuum_well"]:
            well_rule = ex_meta["metrics.vacuum_well"] >= 0
        else:
            return False
        return nfp_rule and well_rule
    ds_meta = ds_meta.filter(criteria_meta)
    count = 0
    valid_ids_meta = []
    for ex in ds_meta:
        valid_ids_meta.append(ex["plasma_config_id"])
        # count = count + 1
        # if count >= run_config.num_config:
        #     break

    print("Number of valid cases:", len(valid_ids_meta))
    
    # Filter using index
    def filter_by_id(ex):
        return ex["plasma_config_id"] in valid_ids_meta # same: replace "id" with the shared key
    
    ds = ds.filter(filter_by_id)

    # Shard the dataset across SLURM array tasks so each worker processes a
    # disjoint slice.  shard(num_shards=N, index=i) selects every N-th record
    # starting at position i, which gives an even split without pre-counting.
    if run_config.num_tasks > 1:
        ds = ds.shard(num_shards=run_config.num_tasks, index=run_config.task_id)
        print(f"Array task {run_config.task_id}/{run_config.num_tasks}: "
              f"processing shard {run_config.task_id} of {run_config.num_tasks}")

    # Each task is responsible for at most ceil(num_config / num_tasks) entries.
    per_task_limit = math.ceil(run_config.num_config / run_config.num_tasks)

    # Dataset iterator
    woutds_iter = iter(ds)

    # Iterating dataset
    count = 0
    for i, t_wout in enumerate(woutds_iter):
        if count > per_task_limit:
            break
        count = count + 1
        print("Plasma config id:", t_wout["plasma_config_id"])
        plasma_config_id = t_wout["plasma_config_id"]
        vmecpp_wout_json = t_wout["json"]
        # Generating folder names        
        this_config_dir = join(SAVE_DIR, f"{plasma_config_id}")
        os.makedirs(this_config_dir, exist_ok=True)
        # ----- Download eq if it is not downloaded already -----
        if Path(this_config_dir + "/quadcoil_inputs.npy").is_file():
            print("Equilibrium downloaded.")
        # Parse the JSON string into a VmecppWOut Pydantic model, then wrap it
        # in a simsopt-compatible equilibrium object so we can access the plasma
        # boundary surface and net current.
        try:
            vmecpp_wout = vmec_utils.VmecppWOut.model_validate_json(vmecpp_wout_json)
            equil = vmec_utils.as_simsopt_vmec(vmecpp_wout)
        except Exception as e:
            # Handle or log the error
            print(f"VMEC wout loading failed: {e}")
            # e.errors() gives a structured list of what failed
            with open(f"{this_config_dir}/validation_error.txt", "w") as text_file:
                text_file.write(str(e))
                text_file.write("---- vmecpp_wout_json -----")
                text_file.write(str(vmecpp_wout_json))
                text_file.write("---- t_wout -----")
                text_file.write(str(t_wout))
            continue
        print("VMEC load succeeded")
        net_poloidal_current_amperes = equil.external_current()
        plasma_surface = equil.boundary
        plasma_surface.change_resolution(
            mpol_max_plasma, ntor_max_plasma
        )
        jnp.save(
            this_config_dir + "/quadcoil_inputs",
            {
                "nfp": run_config.nfp,
                "stellsym": plasma_surface.stellsym,
                "plasma_dofs": plasma_surface.get_dofs(),
                "net_poloidal_current_amperes": net_poloidal_current_amperes,
                "Bnormal_plasma":None,
            }
        )

# ---------------------------------------------------------------------------
# Entry point & argument parsing
# ---------------------------------------------------------------------------
def main():
    import sys    
    
    parser = argparse.ArgumentParser(description="Looking for coils")
    # --device_scaling selects which row of device_scaling_dict to use when
    # computing geometry/field thresholds.
    parser.add_argument("--device_scaling", type=str,
                        help="choose the device scaling among: aries, w7x, stellaris",
                       default="stellaris")
    # --num_config optionally limits how many dataset entries are processed
    # (useful for quick smoke-tests; None means process all).
    parser.add_argument("--num_config", type=int,
                        help="choose the number of configurations to process",
                        default=None)
    
    # --num_config optionally limits how many dataset entries are processed
    # (useful for quick smoke-tests; None means process all).
    parser.add_argument("--nfp", type=int,
                        help="choose the number of field period",
                        default=None)
    # SLURM array task identity.  Defaults fall back to SLURM env vars so the
    # jobscript does not need to pass them explicitly when using --array.
    parser.add_argument("--task_id", type=int,
                        help="0-based index of this task (default: SLURM_ARRAY_TASK_ID or 0)",
                        default=None)
    parser.add_argument("--num_tasks", type=int,
                        help="total number of parallel tasks (default: SLURM_ARRAY_TASK_COUNT or 1)",
                        default=None)

    # When running locally (not on the cluster) inject a default argument so
    # the script works without any command-line flags.
    if not RUNNING_ON_CLUSTER:
        sys.argv.append("--device_scaling=stellaris")

    args = parser.parse_args()

    print("Args loaded:", args)

    if not args.device_scaling:
        parser.print_help()
        exit()

    # Resolve task identity: CLI args > SLURM env vars > single-task defaults.
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = args.num_tasks
    if num_tasks is None:
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    # Store the parsed config in a global so `run()` can access it if needed.
    global run_config
    run_config = RunConfig(args.device_scaling, args.num_config, args.nfp,
                           task_id, num_tasks)
    run(run_config)


class RunConfig:
    """Lightweight container for command-line arguments.

    Stored as a plain object so it can be serialised to JSON via __dict__.
    """
    def __init__(self, device_scaling, num_config, nfp, task_id=0, num_tasks=1):
        self.device_scaling = device_scaling
        self.num_config     = num_config if num_config is not None else 5
        self.nfp            = nfp
        self.task_id        = task_id
        self.num_tasks      = num_tasks

# Entry point for execution
if __name__ == "__main__":
    main()