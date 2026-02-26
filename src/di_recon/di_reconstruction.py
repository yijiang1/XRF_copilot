#!/usr/bin/env python3
"""
Di et al. 2017 XRF tomographic reconstruction — Python/PyTorch implementation.

Algorithm:  Bi-level (outer/inner) optimization
  Outer epoch: freeze self-absorption SA from current W.detach()
  Inner loop:  L-BFGS minimises Poisson NLL (or MSE) over all angles × batches
               gradient flows through InTens and direct W contribution
               SA is frozen → Term-3 gradient is zero (Di et al. approximation)
  After inner: clamp W ≥ 0, advance outer epoch

Reused from src/reconstruction/:
  - MakeFLlinesDictionary_manual() — element physics
  - intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual() — P matrix (n_ranks=1)
  - rotate() — 3D rotation
  - initialize_guess_3d() — initial guess
  - find_lines_roi_idx_from_dataset() — HDF5 channel indexing

Reference:
  Z. Di et al., "Retrieval of few-femtomole analyte in fluorescence
  tomography by supervised machine learning of X-ray fluorescence spectra,"
  Science Advances, 2017.
  MATLAB code: Tao/XRF_XTM_Simulation/
"""

import os
import shutil
import numpy as np
import h5py
import torch as tc
import torch.nn as nn

tc.set_default_dtype(tc.float32)

# ── Reuse from Panpan's reconstruction package ──────────────────────────────
import xraylib as xlib
import xraylib_np as xlib_np

from src.reconstruction.util import (
    MakeFLlinesDictionary_manual,
    rotate,
    intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual,
    find_lines_roi_idx_from_dataset,
)
from src.reconstruction.array_ops import initialize_guess_3d

import warnings
warnings.filterwarnings("ignore")

# Default FL line constants (same as Panpan)
_FL_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE,
                   xlib.KB2_LINE, xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE])
_FL_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE,
                   xlib.LB3_LINE, xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE,
                   xlib.LB7_LINE, xlib.LB9_LINE, xlib.LB10_LINE, xlib.LB15_LINE,
                   xlib.LB17_LINE])
_FL_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])


# ── Loss functions ────────────────────────────────────────────────────────────

def _poisson_nll(pred: tc.Tensor, obs: tc.Tensor, eps: float = 1e-8) -> tc.Tensor:
    """Poisson negative log-likelihood: sum(pred - obs * log(pred + eps)).

    Args:
        pred: predicted photon counts (non-negative).
        obs:  observed photon counts (non-negative).
        eps:  small constant for numerical stability.

    Returns:
        Scalar loss (sum over all elements).
    """
    return (pred - obs * tc.log(pred + eps)).sum()


def _mse(pred: tc.Tensor, obs: tc.Tensor) -> tc.Tensor:
    """Sum-of-squared-errors loss."""
    return ((pred - obs) ** 2).sum()


# ── Self-absorption factor (frozen per outer epoch) ──────────────────────────

def _compute_SA_mb(
    lac_rot: tc.Tensor,
    P_mb: tc.Tensor,
    p: int,
    n_det: int,
    n_element: int,
    n_lines: int,
    n_voxel_mb: int,
    dia_len_n: int,
    dev: tc.device,
) -> tc.Tensor:
    """Compute frozen self-absorption factor for one minibatch at one angle.

    Replicates PPM.init_SA_theta() — called with detached (frozen) lac.

    Args:
        lac_rot: (n_elem, n_lines, n_voxel_mb, n_voxel_total) — frozen, no grad.
                 = W_rot.detach() × FL_line_attCS_ls, expanded.
        P_mb:    (n_det, 3, dia_len_n * n_voxel_mb) path-length tensor.
                 Col 0: source voxel indices, col 1: intersecting voxel indices,
                 col 2: path lengths.
        p:       minibatch index.

    Returns:
        SA_mb: (n_lines, n_voxel_mb) — frozen self-absorption factor, on CPU.
    """
    voxel_idx_offset = p * n_voxel_mb

    # att_exponent: (n_det, n_elem, n_lines, dia_len_n * n_voxel_mb)
    att_exponent = tc.stack([
        lac_rot[
            :, :,
            tc.clamp(P_mb[m, 0] - voxel_idx_offset, 0, n_voxel_mb).to(tc.long),
            P_mb[m, 1].to(tc.long),
        ] * P_mb[m, 2].repeat(n_element, n_lines, 1)
        for m in range(n_det)
    ])

    # sum over intersecting voxels along the FL path
    att_exponent_sum = tc.sum(
        att_exponent.view(n_det, n_element, n_lines, n_voxel_mb, dia_len_n),
        dim=-1,
    )  # (n_det, n_elem, n_lines, n_voxel_mb)

    # average over detector points; sum over elements
    SA_mb = tc.mean(tc.exp(-tc.sum(att_exponent_sum, dim=1)), dim=0)
    # shape: (n_lines, n_voxel_mb)

    return SA_mb.cpu()  # keep on CPU to preserve GPU memory


# ── Differentiable forward pass (one minibatch, one angle) ──────────────────

def _di_forward_mb(
    W: tc.Tensor,
    SA_mb: tc.Tensor,
    theta: tc.Tensor,
    p: int,
    n_element: int,
    n_lines: int,
    n_line_group_each_element: tc.Tensor,
    minibatch_size: int,
    sample_size_n: int,
    sample_size_cm: float,
    probe_intensity: float,
    probe_att: bool,
    probe_attCS_ls: tc.Tensor,
    detected_fl_unit_concentration: tc.Tensor,
    signal_attenuation_factor: float,
    det_solid_angle_ratio: float,
    dev: tc.device,
):
    """Differentiable XRF + XRT forward pass for one minibatch at one angle.

    Gradient flows through:
      - InTens (= exp(−cumsum(W × probe_attCS) × dz))  [Term 2: beam attenuation]
      - fl_map (= W × detected_fl_unit_concentration)    [Term 1: direct FL]
    SA_mb is frozen (Term 3 = 0) — the Di et al. bi-level approximation.

    Args:
        W:       nn.Parameter (n_elem, H, N, N) — full volume, tracked by autograd.
        SA_mb:   (n_lines, n_voxel_mb) — frozen self-absorption factor, CPU tensor.
        theta:   scalar rotation angle in radians.
        p:       minibatch index (0-based).

    Returns:
        xrf_pred: (n_lines, minibatch_size) — predicted XRF per strip.
        xrt_pred: (minibatch_size,) — predicted cumulative attenuation exponent.
    """
    n_voxel_mb = minibatch_size * sample_size_n
    dz = sample_size_cm / sample_size_n

    # Extract minibatch Z-rows from W — maintains gradient
    z_start = (p * minibatch_size) // sample_size_n
    z_end = ((p + 1) * minibatch_size) // sample_size_n
    W_mb = W[:, z_start:z_end, :, :]  # (n_elem, mb_z, N, N) — tracked

    # Rotate W_mb for this projection angle
    W_rot_mb = rotate(W_mb, theta, dev)  # (n_elem, mb_z, N, N) — tracked
    W_rot_mb_3d = W_rot_mb.view(n_element, minibatch_size, sample_size_n)
    W_rot_mb_flat = W_rot_mb_3d.view(n_element, n_voxel_mb)

    # Probe attenuation (InTens) — accumulate cumsum across probe direction (axis 1 = sample_size_n)
    att_exponent_acc_map = tc.zeros(minibatch_size, sample_size_n + 1, device=dev)
    fl_map_tot = tc.zeros(n_lines, n_voxel_mb, device=dev)
    line_idx = 0

    for j in range(n_element):
        if probe_att:
            lac_j = W_rot_mb_3d[j] * probe_attCS_ls[j]  # (mb_z, N) — tracked
            lac_acc = tc.cumsum(lac_j, axis=1)  # (mb_z, N)
            lac_acc = tc.cat(
                [tc.zeros(minibatch_size, 1, device=dev), lac_acc], dim=1
            )  # (mb_z, N+1)
            att_exponent_acc_map = att_exponent_acc_map + lac_acc * dz

        n_lines_j = n_line_group_each_element[j].item()
        fl_unit_j = detected_fl_unit_concentration[line_idx : line_idx + n_lines_j]
        fl_map_j = W_rot_mb_flat[j].unsqueeze(0) * fl_unit_j  # (n_lines_j, n_voxel_mb) — tracked
        fl_map_tot[line_idx : line_idx + n_lines_j] = fl_map_j
        line_idx += n_lines_j

    # InTens — tracked via att_exponent_acc_map
    InTens_flat = tc.exp(-att_exponent_acc_map[:, :-1]).view(n_voxel_mb)

    # XRT prediction: cumulative attenuation exponent along probe path
    xrt_pred = att_exponent_acc_map[:, -1]  # (minibatch_size,) — tracked

    # XRF prediction: probe × FL_emission × SA (SA frozen)
    probe_att_flat = probe_intensity * InTens_flat  # (n_voxel_mb,) — tracked
    SA_mb_dev = SA_mb.to(dev)  # (n_lines, n_voxel_mb) — frozen
    fl_signal = probe_att_flat.unsqueeze(0) * fl_map_tot * SA_mb_dev
    fl_signal = fl_signal * det_solid_angle_ratio * signal_attenuation_factor
    xrf_pred = fl_signal.view(n_lines, minibatch_size, sample_size_n).sum(dim=-1)
    # (n_lines, minibatch_size) — tracked

    return xrf_pred, xrt_pred


# ── Main reconstruction function ─────────────────────────────────────────────

def reconstruct_di_xrftomo(
    # ── Data paths ──
    data_path: str,
    f_XRF_data: str,
    f_XRT_data: str,
    recon_path: str,
    P_folder: str,
    f_P: str = "Intersecting_Length",
    f_recon_grid: str = "di_grid_concentration",
    f_initial_guess: str = "di_initialized_grid_concentration",
    f_recon_parameters: str = "di_recon_parameters.txt",
    # ── Sample geometry ──
    sample_size_n: int = 64,
    sample_height_n: int = 64,
    sample_size_cm: float = 0.01,
    # ── Elements + lines ──
    this_aN_dic: dict = None,                  # e.g. {'Ca': 20, 'Sc': 21}
    element_lines_roi: np.ndarray = None,      # e.g. [['Ca','K'],['Sc','K']]
    n_line_group_each_element: np.ndarray = None,  # e.g. [2, 2]
    # ── Probe ──
    probe_energy: np.ndarray = None,           # shape (1,), keV
    probe_intensity: float = 1.0e7,
    probe_att: bool = True,
    # ── Detector geometry ──
    manual_det_coord: bool = False,
    set_det_coord_cm: np.ndarray = None,
    det_on_which_side: str = "positive",
    manual_det_area: bool = False,
    det_area_cm2: float = None,
    det_dia_cm: float = 0.9,
    det_ds_spacing_cm: float = 0.4,
    det_from_sample_cm: float = 1.6,
    # ── Data indexing ──
    XRT_ratio_dataset_idx: int = 3,
    scaler_counts_us_ic_dataset_idx: int = 1,
    scaler_counts_ds_ic_dataset_idx: int = 2,
    theta_ls_dataset: str = "exchange/theta",
    channel_names: str = "exchange/elements",
    # ── Di et al.-specific ──
    loss_type: str = "poisson",     # "poisson" or "ls"
    beta1_xrt: float = 1.0,         # XRT loss weight
    tikhonov_lambda: float = 0.0,   # L2 regularization weight on W
    n_outer_epochs: int = 5,        # outer bi-level iterations
    lbfgs_n_iter: int = 20,         # inner L-BFGS max function evaluations per step
    lbfgs_history: int = 10,        # L-BFGS memory size
    selfAb: bool = True,
    # ── Init / compute ──
    ini_kind: str = "const",
    init_const: float = 0.0,
    cont_from_check_point: bool = False,
    use_saved_initial_guess: bool = False,
    dev: tc.device = None,
    minibatch_size: int = 64,       # Z-rows per batch (same as Panpan)
    save_every_n_epochs: int = 1,
    progress_callback=None,
    fl_K=None, fl_L=None, fl_M=None,
):
    """Di et al. 2017 XRF tomographic reconstruction.

    Uses PyTorch L-BFGS optimizer with Poisson NLL loss (or MSE) and a
    bi-level outer iteration that freezes self-absorption per epoch.

    Args:
        All parameters match XRFReconstructionParams / Panpan's convention.
        loss_type:        "poisson" — Poisson negative log-likelihood (default);
                          "ls"      — least-squares MSE.
        beta1_xrt:        Weight for XRT fidelity term in joint loss (default 1.0).
        tikhonov_lambda:  Optional L2 regularization on W (default 0.0 = off).
        n_outer_epochs:   Number of outer bi-level iterations (default 5).
        lbfgs_n_iter:     Inner L-BFGS max iterations per outer epoch (default 20).
        lbfgs_history:    L-BFGS memory size (default 10).
    """
    if fl_K is None:
        fl_K = _FL_K
    if fl_L is None:
        fl_L = _FL_L
    if fl_M is None:
        fl_M = _FL_M

    if dev is None:
        dev = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    # ── Derived geometry ──
    dia_len_n = int(1.2 * (sample_height_n**2 + sample_size_n**2 + sample_size_n**2) ** 0.5)
    n_voxel_mb = minibatch_size * sample_size_n
    n_voxel = sample_height_n * sample_size_n ** 2
    n_batch = (sample_height_n * sample_size_n) // minibatch_size

    # ── Output directories ──
    os.makedirs(recon_path, exist_ok=True)
    checkpoint_path = os.path.join(recon_path, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    # ── Load observed data ──
    y1_handle = h5py.File(os.path.join(data_path, f_XRF_data), "r")
    y2_handle = h5py.File(os.path.join(data_path, f_XRT_data), "r")

    # ── Element setup ──
    n_element = len(this_aN_dic)
    aN_ls = np.array(list(this_aN_dic.values()))

    # ── FL line dictionary ──
    fl_all_lines_dic = MakeFLlinesDictionary_manual(
        element_lines_roi,
        n_line_group_each_element,
        probe_energy,
        sample_size_n,
        sample_size_cm,
        fl_line_groups=np.array(["K", "L", "M"]),
        fl_K=fl_K, fl_L=fl_L, fl_M=fl_M,
    )
    n_lines = fl_all_lines_dic["n_lines"]
    FL_line_attCS_ls = tc.as_tensor(
        xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])
    ).float().to(dev)  # (n_elem, n_lines)
    detected_fl_unit_concentration = tc.as_tensor(
        fl_all_lines_dic["detected_fl_unit_concentration"]
    ).float().to(dev)  # (n_lines, n_voxel_mb) — same voxel count convention
    n_line_group_each_element = tc.IntTensor(
        fl_all_lines_dic["n_line_group_each_element"]
    ).to(dev)

    # ── Probe attenuation cross-sections ──
    probe_attCS_ls = tc.as_tensor(
        xlib_np.CS_Total(aN_ls, probe_energy).flatten()
    ).to(dev)  # (n_elem,)

    # ── Rotation angles ──
    theta_ls = tc.from_numpy(
        y1_handle[theta_ls_dataset][...] * np.pi / 180
    ).float()  # (n_theta,) in radians
    n_theta = len(theta_ls)

    # ── XRF + XRT data ──
    element_lines_roi_idx = find_lines_roi_idx_from_dataset(
        data_path, f_XRF_data, element_lines_roi, std_sample=False
    )
    # y1_true: (n_lines, n_theta, H*N) — XRF counts
    y1_true = tc.from_numpy(
        y1_handle["exchange/data"][element_lines_roi_idx]
    ).view(len(element_lines_roi_idx), n_theta, sample_height_n * sample_size_n).to(dev)
    # y2_true: (n_theta, H*N) — negative log transmission
    y2_true = tc.from_numpy(
        y2_handle["exchange/data"][XRT_ratio_dataset_idx]
    ).view(n_theta, sample_height_n * sample_size_n).to(dev)
    y2_true = -tc.log(y2_true.clamp(min=1e-8))

    # ── Detector solid angle ──
    if manual_det_area:
        det_solid_angle_ratio = 1.0
        signal_attenuation_factor = 1.0
    else:
        det_solid_angle_ratio = (np.pi * (det_dia_cm / 2) ** 2) / (
            4 * np.pi * det_from_sample_cm ** 2
        )
        signal_attenuation_factor = 1.0

    # ── P matrix (path lengths for self-absorption) ──
    # Shared with Panpan: if the file exists at P_folder/f_P.h5, reuse it.
    P_save_path = os.path.join(P_folder, f_P)
    if not os.path.isfile(P_save_path + ".h5"):
        print(f"[Di recon] P matrix not found at {P_save_path}.h5 — computing (single rank)...")
        intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(
            n_ranks=1,
            minibatch_size=minibatch_size,
            rank=0,
            manual_det_coord=manual_det_coord,
            set_det_coord_cm=set_det_coord_cm,
            det_on_which_side=det_on_which_side,
            manual_det_area=manual_det_area,
            det_dia_cm=det_dia_cm,
            det_from_sample_cm=det_from_sample_cm,
            det_ds_spacing_cm=det_ds_spacing_cm,
            sample_size_n=sample_size_n,
            sample_size_cm=sample_size_cm,
            sample_height_n=sample_height_n,
            P_folder=P_folder,
            f_P=f_P,
        )
    P_handle = h5py.File(P_save_path + ".h5", "r")
    n_det = P_handle["P_array"].shape[0]

    # ── Initialize W ──
    if cont_from_check_point:
        with h5py.File(os.path.join(recon_path, f_recon_grid + ".h5"), "r") as s:
            W_np = s["sample/densities"][...].astype(np.float32)
        W_data = tc.from_numpy(W_np).to(dev)
    elif use_saved_initial_guess:
        with h5py.File(os.path.join(recon_path, f_initial_guess + ".h5"), "r") as s:
            W_np = s["sample/densities"][...].astype(np.float32)
        W_data = tc.from_numpy(W_np).to(dev)
    else:
        W_data = initialize_guess_3d(
            dev, ini_kind, n_element, sample_size_n, sample_height_n,
            recon_path, f_recon_grid, f_initial_guess, init_const,
        )

    W = nn.Parameter(W_data.clone().detach())  # (n_elem, H, N, N) — tracked

    # Save initial guess
    with h5py.File(os.path.join(recon_path, f_initial_guess + ".h5"), "w") as s:
        grp = s.create_group("sample")
        grp.create_dataset(
            "densities",
            shape=(n_element, sample_height_n, sample_size_n, sample_size_n),
            dtype="f4",
        )
        grp.create_dataset("elements", shape=(n_element,), dtype="S5")
        s["sample/densities"][...] = W.detach().cpu().numpy()
        s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype("S5")
    shutil.copy(
        os.path.join(recon_path, f_initial_guess + ".h5"),
        os.path.join(recon_path, f_recon_grid + ".h5"),
    )

    # ── Write parameter log ──
    with open(os.path.join(recon_path, f_recon_parameters), "w") as fp:
        fp.write(f"method = Di et al. 2017 (Python/PyTorch)\n")
        fp.write(f"n_outer_epochs = {n_outer_epochs}\n")
        fp.write(f"lbfgs_n_iter = {lbfgs_n_iter}\n")
        fp.write(f"lbfgs_history = {lbfgs_history}\n")
        fp.write(f"loss_type = {loss_type}\n")
        fp.write(f"beta1_xrt = {beta1_xrt}\n")
        fp.write(f"tikhonov_lambda = {tikhonov_lambda}\n")
        fp.write(f"selfAb = {selfAb}\n")
        fp.write(f"sample_size_n = {sample_size_n}\n")
        fp.write(f"sample_height_n = {sample_height_n}\n")
        fp.write(f"n_theta = {n_theta}\n")
        fp.write(f"probe_energy_keV = {probe_energy[0]:.3f}\n")
        fp.write(f"probe_intensity = {probe_intensity:.2e}\n")
        fp.write(f"element_lines_roi = {element_lines_roi}\n")

    # ── Choose loss function ──
    _loss_xrf = _poisson_nll if loss_type == "poisson" else _mse
    _loss_xrt = _mse  # XRT always MSE (same as Panpan)

    # ── Outer bi-level loop ──────────────────────────────────────────────────
    XRF_loss_history = []
    XRT_loss_history = []

    for outer_ep in range(n_outer_epochs):
        print(f"[Di recon] Outer epoch {outer_ep + 1}/{n_outer_epochs} — precomputing SA...")

        # ── Pre-compute frozen SA_fixed for all angles × batches ──
        # SA changes with rotation, so one SA set per angle.
        SA_precomputed = {}  # {(angle_idx, batch_p): tensor (n_lines, n_voxel_mb) on CPU}
        with tc.no_grad():
            for angle_idx, theta in enumerate(theta_ls):
                # Rotate detached W and compute lac for SA
                W_rot_det = rotate(W.detach(), theta, dev)
                lac_rot = (
                    W_rot_det.view(n_element, 1, 1, n_voxel)
                    * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
                )  # (n_elem, n_lines, 1, n_voxel) broadcast → expand
                lac_rot = lac_rot.expand(-1, -1, n_voxel_mb, -1).float()
                # (n_elem, n_lines, n_voxel_mb, n_voxel) — frozen

                if selfAb:
                    for p in range(n_batch):
                        P_mb = tc.from_numpy(
                            P_handle["P_array"][
                                :, :,
                                p * dia_len_n * n_voxel_mb : (p + 1) * dia_len_n * n_voxel_mb,
                            ]
                        ).to(dev)
                        SA_mb = _compute_SA_mb(
                            lac_rot, P_mb, p, n_det, n_element, n_lines,
                            n_voxel_mb, dia_len_n, dev,
                        )
                        SA_precomputed[(angle_idx, p)] = SA_mb
                else:
                    for p in range(n_batch):
                        SA_precomputed[(angle_idx, p)] = tc.ones(
                            n_lines, n_voxel_mb, dtype=tc.float32
                        )  # no self-absorption

        print(f"[Di recon] SA precomputed. Starting L-BFGS inner loop...")

        # ── Inner L-BFGS optimizer ──
        optimizer = tc.optim.LBFGS(
            [W],
            lr=1.0,
            history_size=lbfgs_history,
            max_iter=lbfgs_n_iter,
            line_search_fn="strong_wolfe",
        )

        xrf_loss_ep = 0.0
        xrt_loss_ep = 0.0

        def closure():
            nonlocal xrf_loss_ep, xrt_loss_ep
            optimizer.zero_grad()
            total_loss = tc.zeros(1, device=dev)
            xrf_acc = 0.0
            xrt_acc = 0.0

            for angle_idx, theta in enumerate(theta_ls):
                this_theta_idx = angle_idx  # sequential ordering for Di et al.
                for p in range(n_batch):
                    SA_mb = SA_precomputed[(angle_idx, p)]

                    xrf_pred, xrt_pred = _di_forward_mb(
                        W, SA_mb, theta, p,
                        n_element, n_lines, n_line_group_each_element,
                        minibatch_size, sample_size_n, sample_size_cm,
                        probe_intensity, probe_att, probe_attCS_ls,
                        detected_fl_unit_concentration,
                        signal_attenuation_factor, det_solid_angle_ratio, dev,
                    )

                    y1_obs = y1_true[
                        :, this_theta_idx,
                        p * minibatch_size : (p + 1) * minibatch_size,
                    ]  # (n_lines, minibatch_size)
                    y2_obs = y2_true[
                        this_theta_idx,
                        p * minibatch_size : (p + 1) * minibatch_size,
                    ]  # (minibatch_size,)

                    xrf_loss_mb = _loss_xrf(xrf_pred, y1_obs)
                    xrt_loss_mb = _loss_xrt(xrt_pred, y2_obs)
                    total_loss = total_loss + xrf_loss_mb + beta1_xrt * xrt_loss_mb
                    xrf_acc += xrf_loss_mb.item()
                    xrt_acc += xrt_loss_mb.item()

            if tikhonov_lambda > 0:
                total_loss = total_loss + tikhonov_lambda * (W ** 2).sum()

            total_loss.backward()
            xrf_loss_ep = xrf_acc
            xrt_loss_ep = xrt_acc
            return total_loss

        optimizer.step(closure)

        # Non-negativity constraint
        with tc.no_grad():
            W.clamp_(min=0.0)

        XRF_loss_history.append(xrf_loss_ep)
        XRT_loss_history.append(xrt_loss_ep)

        print(
            f"[Di recon] Epoch {outer_ep + 1}: "
            f"XRF_loss={xrf_loss_ep:.4e}, XRT_loss={xrt_loss_ep:.4e}"
        )

        # ── Save checkpoint ──
        W_np = W.detach().cpu().numpy()
        with h5py.File(os.path.join(recon_path, f_recon_grid + ".h5"), "w") as s:
            grp = s.create_group("sample")
            grp.create_dataset(
                "densities",
                shape=(n_element, sample_height_n, sample_size_n, sample_size_n),
                dtype="f4",
            )
            grp.create_dataset("elements", shape=(n_element,), dtype="S5")
            s["sample/densities"][...] = W_np
            s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype("S5")

        if (outer_ep + 1) % save_every_n_epochs == 0:
            ckpt_file = os.path.join(checkpoint_path, f"{f_recon_grid}_{outer_ep}.h5")
            with h5py.File(ckpt_file, "w") as s:
                grp = s.create_group("sample")
                grp.create_dataset(
                    "densities",
                    shape=(n_element, sample_height_n, sample_size_n, sample_size_n),
                    dtype="f4",
                )
                grp.create_dataset("elements", shape=(n_element,), dtype="S5")
                s["sample/densities"][...] = W_np
                s["sample/elements"][...] = np.array(list(this_aN_dic.keys())).astype("S5")

        if progress_callback is not None:
            progress_callback(outer_ep + 1, n_outer_epochs)

        if hasattr(progress_callback, "__self__"):
            # Check stop signal if callback wraps a stop_event check
            pass

    # ── Save loss history ──
    np.save(os.path.join(recon_path, "di_xrf_loss.npy"), np.array(XRF_loss_history))
    np.save(os.path.join(recon_path, "di_xrt_loss.npy"), np.array(XRT_loss_history))

    # ── Close handles ──
    P_handle.close()
    y1_handle.close()
    y2_handle.close()

    recon_file = os.path.join(recon_path, f_recon_grid + ".h5")
    print(f"[Di recon] Done. Result saved to: {recon_file}")
    return recon_file
