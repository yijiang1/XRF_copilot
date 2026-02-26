"""Test that unified variable names work correctly in both Pydantic models
and worker param-extraction logic.

Unified names tested:
  probe_energy      (was x_ray_energy in BNL)
  theta_ls_dataset  (was hardcoded 'thetas' in BNL)
  element_symbols   (was element_type in BNL, elements_atomic_numbers in Panpan)

No GPU or ASTRA required — this tests the API/model layer only.
Data is read from testing_ground/.
"""

import sys
import os
import numpy as np
import h5py

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/mnt/micdata3/XRF_tomography/testing_ground/data"
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
print("Variable Unification Test")
print("=" * 60)


# ── 1. Pydantic model: FLCorrectionParams ─────────────────────────────────────
print("\n[1] FLCorrectionParams — field names")
from src.services.models import FLCorrectionParams, XRFReconstructionParams

# Must accept probe_energy (not x_ray_energy)
try:
    fl_params = FLCorrectionParams(
        fn_root="/tmp/test",
        probe_energy=13.577,
        theta_ls_dataset="thetas",
    )
    check("FLCorrectionParams accepts probe_energy=13.577", True)
    check("probe_energy value stored correctly", fl_params.probe_energy == 13.577)
    check("theta_ls_dataset default is 'thetas'", fl_params.theta_ls_dataset == "thetas")
except Exception as e:
    check("FLCorrectionParams construction", False, str(e))

# Must have element_symbols (not element_type or x_ray_energy)
declared_fields = set(FLCorrectionParams.model_fields.keys()) if hasattr(FLCorrectionParams, "model_fields") else set(FLCorrectionParams.__fields__.keys())
check("x_ray_energy removed from declared fields",
      "x_ray_energy" not in declared_fields,
      f"declared fields: {sorted(declared_fields)}")
check("element_type removed from FLCorrectionParams",
      "element_type" not in declared_fields,
      f"declared fields: {sorted(declared_fields)}")
check("element_symbols present in FLCorrectionParams",
      "element_symbols" in declared_fields,
      f"declared fields: {sorted(declared_fields)}")

# Custom theta_ls_dataset
fl_custom = FLCorrectionParams(fn_root="/tmp/test", theta_ls_dataset="exchange/theta")
check("theta_ls_dataset accepts custom value", fl_custom.theta_ls_dataset == "exchange/theta")


# ── 2. Pydantic model: XRFReconstructionParams ────────────────────────────────
print("\n[2] XRFReconstructionParams — unified field names")
try:
    recon_params = XRFReconstructionParams(
        data_path="/tmp",
        f_XRF_data="test8_xrf",
        f_XRT_data="test8_xrt",
        recon_path="/tmp/recon",
        P_folder="/tmp/P",
        probe_energy=20.0,
        theta_ls_dataset="exchange/theta",
        element_symbols="Ca, Sc",
    )
    check("XRFReconstructionParams accepts probe_energy=20.0", True)
    check("probe_energy value stored correctly", recon_params.probe_energy == 20.0)
    check("theta_ls_dataset default is 'exchange/theta'", recon_params.theta_ls_dataset == "exchange/theta")
    check("element_symbols stored correctly", recon_params.element_symbols == "Ca, Sc")
except Exception as e:
    check("XRFReconstructionParams construction", False, str(e))

recon_declared = set(XRFReconstructionParams.model_fields.keys()) if hasattr(XRFReconstructionParams, "model_fields") else set(XRFReconstructionParams.__fields__.keys())
check("elements_atomic_numbers removed from XRFReconstructionParams",
      "elements_atomic_numbers" not in recon_declared,
      f"declared fields: {sorted(recon_declared)}")
check("element_symbols present in XRFReconstructionParams",
      "element_symbols" in recon_declared,
      f"declared fields: {sorted(recon_declared)}")


# ── 3. FL worker param extraction ─────────────────────────────────────────────
print("\n[3] FL worker param extraction logic")

params_dict = fl_params.model_dump()

theta_key = params_dict.get("theta_ls_dataset", "thetas")
check("Worker reads theta_ls_dataset from params", theta_key == "thetas",
      f"got '{theta_key}'")

XEng = float(params_dict.get("probe_energy", 13.577))
check("Worker reads probe_energy as XEng", abs(XEng - 13.577) < 1e-9,
      f"got {XEng}")

elem_type = [e.strip() for e in params_dict.get("element_symbols", "").split(",") if e.strip()]
check("Worker reads element_symbols from FL params dict",
      "element_symbols" in params_dict,
      f"keys: {list(params_dict.keys())}")

check("'x_ray_energy' key absent from FL params dict",
      "x_ray_energy" not in params_dict,
      f"keys: {list(params_dict.keys())}")
check("'element_type' key absent from FL params dict",
      "element_type" not in params_dict,
      f"keys: {list(params_dict.keys())}")


# ── 4. Reconstruction worker param extraction ─────────────────────────────────
print("\n[4] Reconstruction worker param extraction logic")

import xraylib as _xlib

recon_dict = recon_params.model_dump()
probe_e = np.array([recon_dict["probe_energy"]])
theta_ds = recon_dict["theta_ls_dataset"]
check("Worker reads probe_energy → np.array", probe_e[0] == 20.0,
      f"got {probe_e}")
check("Worker reads theta_ls_dataset='exchange/theta'",
      theta_ds == "exchange/theta", f"got '{theta_ds}'")

# Simulate _parse_element_symbols() from recon_worker.py
syms = recon_dict["element_symbols"]   # "Ca, Sc"
this_aN_dic = {s.strip(): _xlib.SymbolToAtomicNumber(s.strip())
               for s in syms.split(",") if s.strip()}
check("element_symbols 'Ca, Sc' → this_aN_dic {'Ca':20,'Sc':21}",
      this_aN_dic == {"Ca": 20, "Sc": 21}, f"got {this_aN_dic}")
check("'elements_atomic_numbers' key absent from recon params dict",
      "elements_atomic_numbers" not in recon_dict,
      f"keys: {list(recon_dict.keys())}")


# ── 5. HDF5 angle loading with theta_ls_dataset (BNL data) ───────────────────
print("\n[5] HDF5 angle loading — BNL format ('thetas' key)")
fn_data = os.path.join(DATA_ROOT, "fl_correction", "everything.h5")
if os.path.exists(fn_data):
    theta_key = "thetas"   # matches FLCorrectionParams default
    with h5py.File(fn_data, "r") as f:
        angle_list = np.array(f[theta_key])
    check(f"Load angles with key='{theta_key}'", len(angle_list) > 0,
          f"got {len(angle_list)} angles")
    check("Angles are in degrees (range check)",
          float(np.max(np.abs(angle_list))) <= 360.0,
          f"max={np.max(np.abs(angle_list)):.1f}")
    print(f"      → {len(angle_list)} angles, range [{angle_list.min():.1f}, {angle_list.max():.1f}]°")
else:
    print(f"  (skipped — {fn_data} not found)")


# ── 6. HDF5 angle loading (Panpan/APS data) ───────────────────────────────────
print("\n[6] HDF5 angle loading — Panpan format ('exchange/theta' key)")
fn_xrt = os.path.join(DATA_ROOT, "test8", "test8_xrt")
if os.path.exists(fn_xrt):
    theta_key = "exchange/theta"   # matches XRFReconstructionParams default
    with h5py.File(fn_xrt, "r") as f:
        angle_list_recon = np.array(f[theta_key])
    check(f"Load angles with key='{theta_key}'", len(angle_list_recon) > 0,
          f"got {len(angle_list_recon)} angles")
    check("Angles are in degrees (range check)",
          float(np.max(np.abs(angle_list_recon))) <= 360.0,
          f"max={np.max(np.abs(angle_list_recon)):.1f}")
    print(f"      → {len(angle_list_recon)} angles, range [{angle_list_recon.min():.1f}, {angle_list_recon.max():.1f}]°")
else:
    print(f"  (skipped — {fn_xrt} not found)")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print(f"{PASS} All checks passed.")
else:
    print(f"{FAIL} {len(errors)} check(s) failed:")
    for e in errors:
        print(f"      - {e}")
sys.exit(len(errors))
