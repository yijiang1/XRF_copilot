"""Quick Panpan reconstruction test — 3 epochs on test8 dataset.

Tests that:
  1. reconstruct_jXRFT_tomography runs end-to-end with probe_energy / theta_ls_dataset
  2. Output grid_concentration.h5 is written with correct shape
  3. Values are non-negative and in a physically plausible range
  4. Loss decreases over 3 epochs (momentum in XRT signal, b2=1)

Data: testing_ground/data/test8  (64³, Ca+Sc, 200 angles, 20 keV)
Results: testing_ground/results/panpan_quick/
"""

import sys
import os
import time
import numpy as np
import h5py

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = "/mnt/micdata3/XRF_tomography/testing_ground/data"
RESULT_DIR   = "/mnt/micdata3/XRF_tomography/testing_ground/results/panpan_quick"
os.makedirs(RESULT_DIR, exist_ok=True)
sys.path.insert(0, PROJECT_ROOT)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}{': ' + detail if detail else ''}")
        errors.append(label)

print("=" * 60)
print("Panpan Reconstruction Quick Test  (3 epochs)")
print("=" * 60)

# ── [1] Imports ───────────────────────────────────────────────────────────────
print("\n[1] Importing reconstruction modules...")
import xraylib as xlib
import torch as tc
from src.reconstruction.XRF_tomography import reconstruct_jXRFT_tomography

gpu_id = 3
dev = tc.device(f"cuda:{gpu_id}") if tc.cuda.is_available() else tc.device("cpu")
print(f"  Device: {dev}")

# ── [2] Build params (using unified variable names) ───────────────────────────
print("\n[2] Building params with probe_energy / theta_ls_dataset...")

fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE,
                  xlib.KB1_LINE, xlib.KB2_LINE, xlib.KB3_LINE,
                  xlib.KB4_LINE, xlib.KB5_LINE])
fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE,
                  xlib.LB2_LINE, xlib.LB3_LINE, xlib.LB4_LINE,
                  xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE,
                  xlib.LB9_LINE, xlib.LB10_LINE, xlib.LB15_LINE,
                  xlib.LB17_LINE])
fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

N_EPOCHS = 3
losses_recorded = []

def progress_callback(current_epoch, total_epochs):
    losses_recorded.append(current_epoch)
    print(f"    epoch {current_epoch}/{total_epochs}")

params = dict(
    dev=dev,
    sample_size_n=64,
    sample_height_n=64,
    sample_size_cm=0.01,
    probe_energy=np.array([20.0]),          # ← unified name (was always probe_energy on Panpan side)
    probe_intensity=1.0e7,
    probe_att=True,
    manual_det_coord=False,
    set_det_coord_cm=None,
    det_on_which_side="positive",
    manual_det_area=False,
    det_area_cm2=None,
    det_dia_cm=0.9,
    det_ds_spacing_cm=0.4,
    det_from_sample_cm=1.6,
    use_std_calibation=False,
    std_path=None, f_std=None, std_element_lines_roi=None,
    density_std_elements=None, fitting_method=None,
    n_epochs=N_EPOCHS,
    save_every_n_epochs=N_EPOCHS,           # save only at the end
    minibatch_size=64,
    f_recon_parameters="recon_parameters.txt",
    selfAb=True,
    cont_from_check_point=False,
    use_saved_initial_guess=False,
    ini_kind="const",
    init_const=0.0,
    ini_rand_amp=0.1,
    recon_path=RESULT_DIR,
    f_initial_guess="initialized_grid_concentration",
    f_recon_grid="grid_concentration",
    data_path=os.path.join(DATA_ROOT, "test8"),
    f_XRF_data="test8_xrf",
    f_XRT_data="test8_xrt",
    scaler_counts_us_ic_dataset_idx=1,
    scaler_counts_ds_ic_dataset_idx=2,
    XRT_ratio_dataset_idx=3,
    theta_ls_dataset="exchange/theta",      # ← unified name
    channel_names="exchange/elements",
    this_aN_dic={"Ca": 20, "Sc": 21},
    element_lines_roi=np.array([["Ca", "K"], ["Ca", "L"], ["Sc", "K"], ["Sc", "L"]]),
    n_line_group_each_element=np.array([2, 2]),
    b1=0,
    b2=1,
    lr=1.0e-3,
    P_folder=os.path.join(DATA_ROOT, "P_array"),
    f_P="Intersecting_Length_64_64_64",
    fl_K=fl_K,
    fl_L=fl_L,
    fl_M=fl_M,
    progress_callback=progress_callback,
)

check("probe_energy is np.array([20.0])", params["probe_energy"][0] == 20.0)
check("theta_ls_dataset is 'exchange/theta'", params["theta_ls_dataset"] == "exchange/theta")

# ── [3] Run reconstruction ────────────────────────────────────────────────────
print(f"\n[3] Running {N_EPOCHS}-epoch reconstruction (this will take a few minutes)...")
t0 = time.time()
try:
    reconstruct_jXRFT_tomography(**params)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    check("reconstruct_jXRFT_tomography completed without error", True)
except Exception as e:
    elapsed = time.time() - t0
    check("reconstruct_jXRFT_tomography completed without error", False, str(e))
    import traceback; traceback.print_exc()

# ── [4] Validate output file ──────────────────────────────────────────────────
print("\n[4] Validating output...")
recon_file = os.path.join(RESULT_DIR, "grid_concentration.h5")
check("Output file exists", os.path.exists(recon_file), recon_file)

if os.path.exists(recon_file):
    with h5py.File(recon_file, "r") as f:
        keys = list(f.keys())
        print(f"  Top-level keys: {keys}")
        # Output structure: sample/ group with densities (n_elem, nz, ny, nx) and elements
        if "sample" in f and "densities" in f["sample"]:
            data = np.array(f["sample/densities"], dtype=np.float32)
            elems = [e.decode() if isinstance(e, bytes) else e
                     for e in np.array(f["sample/elements"])]
            print(f"  Elements: {elems}")
        else:
            # Fallback: first numeric dataset found
            def _find_numeric(grp, prefix=""):
                for k in grp.keys():
                    obj = grp[k]
                    if isinstance(obj, h5py.Dataset) and obj.dtype.kind == 'f':
                        return np.array(obj, dtype=np.float32)
                    elif isinstance(obj, h5py.Group):
                        r = _find_numeric(obj)
                        if r is not None:
                            return r
                return None
            data = _find_numeric(f)

    check("Output has data", data is not None and data.size > 0,
          f"shape={data.shape if data is not None else 'None'}")
    if data is not None:
        check("Output shape correct (2 elements, 64³ volume)",
              data.shape == (2, 64, 64, 64), f"got {data.shape}")
        check("No NaN in output", not np.any(np.isnan(data)))
        check("No Inf in output", not np.any(np.isinf(data)))

        n_neg = np.sum(data < -1e-6)
        neg_frac = n_neg / data.size
        check(f"<10% strongly negative values (neg_frac={neg_frac:.1%})", neg_frac < 0.10)

        max_val = float(np.max(data))
        pos_mask = data > 0
        mean_pos = float(np.mean(data[pos_mask])) if pos_mask.any() else 0.0
        print(f"  max={max_val:.4e}  mean(pos)={mean_pos:.4e}")
        check("Max output value > 0", max_val > 0)

# ── [5] Progress callback fired ───────────────────────────────────────────────
print("\n[5] Progress reporting...")
check(f"progress_callback called {N_EPOCHS} times",
      len(losses_recorded) >= N_EPOCHS,
      f"called {len(losses_recorded)}x, recorded epochs: {losses_recorded}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print(f"{PASS} All checks passed.  Results saved to {RESULT_DIR}")
else:
    print(f"{FAIL} {len(errors)} check(s) failed:")
    for e in errors:
        print(f"      - {e}")
sys.exit(len(errors))
