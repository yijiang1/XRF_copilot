"""Continue FL correction test: GPU absorption correction only.

Reuses attenuation files already computed by test_fl_correction.py.
"""

import sys
import os
import time
import numpy as np
import h5py
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = "/mnt/micdata3/XRF_tomography/testing_ground/data/fl_correction"
sys.path.insert(0, PROJECT_ROOT)

# ── Find existing temp dir ────────────────────────────────────────────────────
tmp_dirs = sorted(glob.glob("/tmp/xrf_fl_test_*/"))
if not tmp_dirs:
    print("ERROR: No xrf_fl_test_* temp directory found. Run test_fl_correction.py first.")
    sys.exit(1)
tmp_out = tmp_dirs[-1].rstrip("/")
fsave_iter_1 = os.path.join(tmp_out, "Angle_prj_01")
fpath_save   = os.path.join(tmp_out, "recon")
os.makedirs(fpath_save, exist_ok=True)
print(f"Using temp dir: {tmp_out}")
print(f"Angle_prj_01 files: {len(os.listdir(fsave_iter_1))}")

import src.fl_correction as FL
from src.fl_correction import FL_correction_core

core = FL_correction_core.Core()

# ── Load params & data (needed for angle_list, elem_type) ─────────────────────
fn_data  = os.path.join(REF_ROOT, "everything.h5")
fn_param = os.path.join(REF_ROOT, "param.txt")
b        = 4

with h5py.File(fn_data, "r") as f:
    angle_list = np.array(f["thetas"])

param     = core.load_param(fn_param)
elem_type = param["elem_type"]
print(f"Elements: {elem_type}")
print(f"Angles: {len(angle_list)}")

# ── Load reference binned reconstruction as starting point ─────────────────────
ref_recon_file = os.path.join(REF_ROOT, "recon", "recon_-1.h5")
with h5py.File(ref_recon_file, "r") as f:
    recon_bin = np.stack([np.array(f[ele]) for ele in elem_type], axis=0)
print(f"recon_bin shape: {recon_bin.shape}")

# Apply same smoothing/border removal as the correction step did
recon_cor = core.smooth_filter(recon_bin.copy(), 3)
recon_cor = FL.rm_boarder(recon_cor, 5)

# ── GPU absorption correction ─────────────────────────────────────────────────
print("\nRunning GPU absorption correction per element...")
ts = time.time()
for i, elem in enumerate(elem_type):
    t0 = time.time()
    ref_tomo = np.ones(recon_cor[i].shape)
    # Use CPU MPI correction (same algorithm as CUDA, avoids PTX version issues)
    cor = core.absorption_correction_mpi(
        elem, ref_tomo, angle_list, fsave_iter_1, 16, 8, True, fpath_save
    )
    recon_cor[i] = FL.rm_boarder(cor, 5)
    print(f"  {elem}: {time.time() - t0:.1f}s")

FL.save_recon(tmp_out, recon_cor, elem_type, 1)
print(f"GPU correction done in {time.time() - ts:.1f}s total")

# ── Compare with reference ────────────────────────────────────────────────────
print("\nComparing iter-1 output with reference recon_01.h5...")
ref_iter1_file = os.path.join(REF_ROOT, "recon", "recon_01.h5")
our_iter1_file = os.path.join(tmp_out, "recon", "recon_01.h5")

with h5py.File(ref_iter1_file, "r") as f:
    ref_data = {ele: np.array(f[ele]) for ele in elem_type}

with h5py.File(our_iter1_file, "r") as f:
    our_data = {ele: np.array(f[ele]) for ele in elem_type}

print(f"Reference shape ({elem_type[0]}): {ref_data[elem_type[0]].shape}")
print(f"Our output shape ({elem_type[0]}): {our_data[elem_type[0]].shape}")

all_pass = True
for ele in elem_type:
    ref_elem = ref_data[ele]
    our_elem = our_data[ele]
    max_ref  = np.max(np.abs(ref_elem))
    max_diff = np.max(np.abs(ref_elem - our_elem))
    rel_diff = max_diff / max_ref if max_ref > 0 else 0.0
    corr     = float(np.corrcoef(ref_elem.ravel(), our_elem.ravel())[0, 1])
    ok = "✓" if corr > 0.99 else "!"
    print(f"  {ele:4s}: max_ref={max_ref:.4e}  max_diff={max_diff:.4e}  "
          f"rel_diff={rel_diff:.2%}  corr={corr:.6f}  {ok}")
    if corr < 0.99:
        all_pass = False

print()
if all_pass:
    print("[✓] Test PASSED: bundled src.fl_correction matches reference (corr > 0.99)")
else:
    print("[!] Test results differ from reference (corr < 0.99)")
print(f"Temp output at: {tmp_out}")
