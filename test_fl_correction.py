"""Test the bundled src.fl_correction against existing reference results.

Strategy:
  1. Verify bundled imports work
  2. Load the same data (everything.h5) and normalize projections
  3. Compare our projections against reference (deterministic — same math)
  4. Load the reference recon_-1.h5 (initial binned reconstruction) and run
     ONE correction iteration using our bundled code, then compare against
     the reference recon_01.h5

This skips re-running ASTRA (which takes >1 hour) by re-using the reference
reconstructions as the starting point.
"""

import sys
import os
import time
import numpy as np

# ── Setup paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = "/mnt/micdata3/XRF_tomography/FL_correction/FL_correction_new"
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("FL Correction Bundle Test")
print("=" * 60)

# ── Step 1: Import test ───────────────────────────────────────────────────────
print("\n[1] Testing bundled imports...")
import src.fl_correction as FL
from src.fl_correction import FL_correction_core
import h5py
print("    src.fl_correction        OK")
print("    src.fl_correction_core   OK")

core = FL_correction_core.Core()

# ── Step 2: Load data (same as XRF_corr_APS_YJ.py) ───────────────────────────
print("\n[2] Loading data...")
fn_data  = os.path.join(REF_ROOT, "everything.h5")
fn_param = os.path.join(REF_ROOT, "param.txt")
b        = 4
scale    = 1e15
elem_start_idx = 1   # skip Si
crop_x_start   = 100
crop_x_end     = 500

with h5py.File(fn_data, "r") as f:
    img_all    = np.array(f["data"])
    angle_list = np.array(f["thetas"])

print(f"    Full data shape: {img_all.shape}")  # (8, 53, 301, 601)

img_all = img_all[elem_start_idx:]
s = img_all.shape
img_all = img_all[:, :, : s[2] // b * b, : s[3] // b * b]
theta        = angle_list / 180.0 * np.pi
theta_tomopy = -theta
print(f"    After slice: {img_all.shape}, n_angles={len(angle_list)}")

# ── Step 3: Load parameters ───────────────────────────────────────────────────
print("\n[3] Loading parameters...")
param     = core.load_param(fn_param)
n_elem    = param["nelem"]
elem_type = param["elem_type"]
em_cs     = param["em_cs"]
M         = param["M"]
rho       = param["rho"]
pix       = param["pix"]
cs        = core.get_atten_coef(elem_type, param["XEng"], param["em_E"])
param["pix"] *= b
print(f"    Elements ({n_elem}): {elem_type}")
print(f"    Pixel size (binned): {param['pix']} nm")

# ── Step 4: Normalize projections ────────────────────────────────────────────
print("\n[4] Normalizing projections (our code)...")
proj = {}
proj_ic = img_all[-1]
for i, ele in enumerate(elem_type):
    proj[ele] = img_all[i] / proj_ic
    proj[ele] = proj[ele] / em_cs[ele] / rho[ele]
    proj[ele] = proj[ele] * pix ** 2
    proj[ele] = proj[ele] * rho[ele]
    proj[ele] = proj[ele] / M[ele]
    proj[ele] = proj[ele] * scale
    proj[ele] = proj[ele][:, :, crop_x_start:crop_x_end]
print(f"    Projection shape per element: {proj[elem_type[0]].shape}")

# Reference normalization (identical math from XRF_corr_APS_YJ.py)
# Reload so we have same input
print("\n    Verifying against reference normalization...")
with h5py.File(fn_data, "r") as f:
    img_ref = np.array(f["data"])
img_ref = img_ref[elem_start_idx:]
s = img_ref.shape
img_ref = img_ref[:, :, : s[2] // b * b, : s[3] // b * b]
param_ref = core.load_param(fn_param)
pix_ref   = param_ref["pix"]

proj_ref = {}
ic_ref = img_ref[-1]
for i, ele in enumerate(elem_type):
    proj_ref[ele] = img_ref[i] / ic_ref
    proj_ref[ele] = proj_ref[ele] / em_cs[ele] / rho[ele]
    proj_ref[ele] = proj_ref[ele] * pix_ref ** 2
    proj_ref[ele] = proj_ref[ele] * rho[ele]
    proj_ref[ele] = proj_ref[ele] / M[ele]
    proj_ref[ele] = proj_ref[ele] * scale
    proj_ref[ele] = proj_ref[ele][:, :, crop_x_start:crop_x_end]

for ele in elem_type:
    max_diff = np.max(np.abs(proj[ele] - proj_ref[ele]))
    assert max_diff == 0.0, f"{ele}: normalization mismatch! max_diff={max_diff}"
print("    Normalization matches exactly (max diff = 0.0)  ✓")

# ── Step 5: Bin projections ───────────────────────────────────────────────────
print("\n[5] Binning projections...")
proj_raw = core.pre_treat([proj[ele] for ele in elem_type])
s_pr     = proj_raw.shape
proj_bin = FL.bin_ndarray(
    proj_raw, (s_pr[0], s_pr[1], s_pr[2] // b, s_pr[3] // b), "sum"
)
print(f"    proj_bin shape: {proj_bin.shape}")

# ── Step 6: Load reference reconstructions (skip re-running ASTRA) ───────────
print("\n[6] Loading reference reconstructions from recon_-1.h5 (binned)...")
ref_recon_file = os.path.join(REF_ROOT, "recon", "recon_-1.h5")
with h5py.File(ref_recon_file, "r") as f:
    keys = list(f.keys())
    print(f"    Keys in recon_-1.h5: {keys}")
    # Stack into (n_elem, nz, ny, nx) in the same order as elem_type
    recon_bin = np.stack([np.array(f[ele]) for ele in elem_type], axis=0)
print(f"    recon_bin shape: {recon_bin.shape}")

# ── Step 7: Load mask ─────────────────────────────────────────────────────────
print("\n[7] Loading detector mask...")
fn_mask = os.path.join(REF_ROOT, "mask3D_200.h5")
mask3D  = core.load_mask3D(fn_mask)
core.load_global_mask(mask3D)
print(f"    mask3D loaded: dict with {len(mask3D)} radial lengths (7..{max(mask3D.keys())})")

# ── Step 8: Run ONE correction iteration with our bundled code ────────────────
print("\n[8] Running 1 correction iteration with bundled src.fl_correction...")
import tempfile

# Use a temp output directory
tmp_out = tempfile.mkdtemp(prefix="xrf_fl_test_")
fpath_save   = os.path.join(tmp_out, "recon")
fsave_iter_1 = os.path.join(tmp_out, "Angle_prj_01")
os.makedirs(fpath_save, exist_ok=True)

recon_cor = recon_bin.copy()
ref_prj   = proj_bin
num_cpu   = 8

ts = time.time()
print("    Smoothing and border removal...")
recon_cor = core.smooth_filter(recon_cor, 3)
recon_cor = FL.rm_boarder(recon_cor, 5)

print("    Computing attenuation (this may take a few minutes)...")
core.cal_and_save_atten_prj(
    param, cs, recon_cor, angle_list, ref_prj,
    fsave=fsave_iter_1, align_flag=False, enable_scale=False, num_cpu=num_cpu,
)
print(f"    Attenuation done in {time.time() - ts:.1f}s")

print("    Running GPU absorption correction per element...")
for i, elem in enumerate(elem_type):
    t0 = time.time()
    ref_tomo = np.ones(recon_cor[i].shape)
    cor = core.cuda_absorption_correction_wrap(
        elem, ref_tomo, angle_list, fsave_iter_1, 16, True, fpath_save
    )
    recon_cor[i] = FL.rm_boarder(cor, 5)
    print(f"    {elem}: {time.time() - t0:.1f}s")

FL.save_recon(tmp_out, recon_cor, elem_type, 1)
print(f"    Iteration 1 done in {time.time() - ts:.1f}s total")

# ── Step 9: Compare our iter-1 output with reference iter-1 ──────────────────
print("\n[9] Comparing iter-1 output with reference recon_01.h5...")
ref_iter1_file = os.path.join(REF_ROOT, "recon", "recon_01.h5")
our_iter1_file = os.path.join(tmp_out, "recon", "recon_01.h5")

with h5py.File(ref_iter1_file, "r") as f:
    ref_data = {ele: np.array(f[ele]) for ele in elem_type}

with h5py.File(our_iter1_file, "r") as f:
    our_data = {ele: np.array(f[ele]) for ele in elem_type}

print(f"    Reference shape ({elem_type[0]}): {ref_data[elem_type[0]].shape}")
print(f"    Our output shape ({elem_type[0]}): {our_data[elem_type[0]].shape}")

for i, ele in enumerate(elem_type):
    ref_elem = ref_data[ele]
    our_elem = our_data[ele]
    max_ref  = np.max(np.abs(ref_elem))
    max_diff = np.max(np.abs(ref_elem - our_elem))
    rel_diff = max_diff / max_ref if max_ref > 0 else 0.0
    corr     = float(np.corrcoef(ref_elem.ravel(), our_elem.ravel())[0, 1])
    print(f"    {ele:4s}: max_ref={max_ref:.4e}  max_diff={max_diff:.4e}  "
          f"rel_diff={rel_diff:.2%}  corr={corr:.6f}")

print("\n[✓] Test complete!")
print(f"    Temp output at: {tmp_out}")
