"""Worker process for FL self-absorption correction (BNL).

Runs in a multiprocessing.Process spawned by fl_run.py.
Reports progress via status_queue; supports stop via stop_event.
"""

import os
import sys
import time
import logging

logger = logging.getLogger(__name__)


def fl_correction_worker_process(params: dict, status_queue, stop_event):
    """Run FL self-absorption correction in-process.

    Args:
        params: FL correction parameters dict.
        status_queue: multiprocessing.Queue for progress/log/result updates.
        stop_event: multiprocessing.Event; set to request cancellation.
    """
    def _put_log(message, level="INFO"):
        status_queue.put({"log": message, "level": level, "timestamp": time.time()})

    def _put_progress(current_step, total_steps, label=""):
        status_queue.put({
            "current_step": current_step,
            "total_steps": total_steps,
            "step_label": label,
        })

    def _check_stop():
        if stop_event.is_set():
            raise RuntimeError("FL correction stopped by user.")

    try:
        import numpy as np
        import h5py

        # Add project root to path so src.* imports work (same as recon_worker.py)
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        import src.fl_correction as FL
        from src.fl_correction import FL_correction_core

    except ImportError as e:
        _put_log(
            f"Import error: {e}. Make sure astra, tomopy, and numba "
            "are installed in the xrf_copilot environment.",
            level="ERROR",
        )
        status_queue.put({"error": str(e)})
        status_queue.put({"finished": True})
        return

    try:
        # ── Parse params ─────────────────────────────────────────────────────
        fn_root        = params["fn_root"]
        fn_data        = params.get("fn_data", "everything.h5")
        b              = int(params.get("binning_factor", 4))
        scale          = float(params.get("scale", 1e15))
        ic_channel_idx = int(params.get("ic_channel_idx", -1))
        elem_ch_indices_raw = params.get("element_channel_indices", "")
        crop_x_start   = int(params.get("crop_x_start", 0))
        crop_x_end     = int(params.get("crop_x_end", -1))    # -1 = no crop
        crop_y_start   = int(params.get("crop_y_start", 0))
        crop_y_end     = int(params.get("crop_y_end", -1))    # -1 = no crop
        det_alfa       = float(params.get("det_alfa", 20.6))
        det_theta      = float(params.get("det_theta", 20.6))
        mask_length    = int(params.get("mask_length_maximum", 200))
        recon_method   = params.get("recon_method", "EM_CUDA")
        recon_n_iter   = int(params.get("recon_n_iter", 16))
        n_corr_iters   = int(params.get("n_correction_iters", 4))
        corr_n_iter    = int(params.get("correction_n_iter", 16))
        border_pixels  = int(params.get("border_pixels", 5))
        smooth_size    = int(params.get("smooth_filter_size", 3))
        num_cpu        = int(params.get("num_cpu", 8))
        use_gpu        = bool(params.get("use_gpu", True))

        if not os.path.isabs(fn_data):
            fn_data = os.path.join(fn_root, fn_data)

        core = FL_correction_core.Core()

        # ── Step 1: Load data ─────────────────────────────────────────────────
        _check_stop()
        _put_log(f"Loading XRF data from {fn_data}")
        with h5py.File(fn_data, "r") as f:
            img_all    = np.array(f["data"])
            theta_key  = params.get("theta_ls_dataset", "thetas")
            angle_list = np.array(f[theta_key])

        s = img_all.shape
        img_all = img_all[:, :, : s[2] // b * b, : s[3] // b * b]
        theta = angle_list / 180.0 * np.pi
        theta_tomopy = -theta
        _put_log(f"Data shape after slicing: {img_all.shape}, angles: {len(angle_list)}")

        # ── Step 2: Build param dict from GUI inputs ───────────────────────────
        _check_stop()
        import xraylib
        XEng       = float(params.get("probe_energy", 13.577))
        pix_nm     = float(params.get("pixel_size_nm", 500))
        pix        = float(f"{pix_nm * 1e-7:.1e}")  # nm → cm, same rounding as load_param
        elem_type  = [e.strip() for e in params.get("element_symbols", "Ti, Cr, Fe, Ni, Ba").split(",") if e.strip()]
        xrf_shell  = [s.strip() for s in params.get("xrf_shell", "K, K, K, K, L").split(",") if s.strip()]
        rho_values = [float(x.strip()) for x in params.get("density", "4.506, 7.19, 7.874, 8.90, 3.59").split(",") if x.strip()]
        em_E_vals  = [float(x.strip()) for x in params.get("emission_energy", "4.509, 5.411, 6.399, 7.472, 4.463").split(",") if x.strip()]
        n_elem     = len(elem_type)
        # Absolute channel indices for each element (from checkbox selection).
        # Falls back to sequential 0..n_elem-1 if not provided (legacy mode).
        elem_ch_indices_str = params.get("element_channel_indices", elem_ch_indices_raw)
        if elem_ch_indices_str.strip():
            elem_ch_indices = [int(x.strip()) for x in elem_ch_indices_str.split(",") if x.strip()]
        else:
            elem_ch_indices = list(range(n_elem))

        M, em_E, em_cs, rho = {}, {}, {}, {}
        for i, ele in enumerate(elem_type):
            atom_idx = xraylib.SymbolToAtomicNumber(ele)
            M[ele]    = xraylib.AtomicWeight(atom_idx)
            em_E[ele] = em_E_vals[i]
            rho[ele]  = rho_values[i]
            shell = xrf_shell[i]
            if shell == "K":
                em_cs[ele]  = xraylib.CS_FluorLine(atom_idx, xraylib.KA_LINE, XEng)
                em_cs[ele] += xraylib.CS_FluorLine(atom_idx, xraylib.KB_LINE, XEng)
            elif shell == "L":
                em_cs[ele]  = xraylib.CS_FluorLine(atom_idx, xraylib.LA_LINE, XEng)
                em_cs[ele] += xraylib.CS_FluorLine(atom_idx, xraylib.LB_LINE, XEng)
            else:
                raise ValueError(f"Unknown XRF shell '{shell}' for element {ele}; expected K or L")

        param = {
            "XEng":      XEng,
            "nelem":     n_elem,
            "rho":       rho,
            "pix":       pix,
            "M":         M,
            "em_E":      em_E,
            "em_cs":     em_cs,
            "elem_type": elem_type,
        }
        cs = core.get_atten_coef(elem_type, XEng, em_E)
        param["pix"] *= b  # scale pixel size for binned volume
        _put_log(f"Elements: {elem_type}, XEng={XEng} keV, pix={pix:.1e} cm (×{b}={param['pix']:.1e} cm)")

        total_steps  = 2 + 1 + 1 + n_elem + 1 + 1 + n_corr_iters * (1 + n_elem)
        current_step = 0

        _put_progress(current_step, total_steps, "Loaded data and parameters")

        # ── Step 3: Normalize projections ─────────────────────────────────────
        _check_stop()
        current_step += 2
        _put_progress(current_step, total_steps, "Normalizing projections...")

        proj = {}
        proj_ic = img_all[ic_channel_idx]
        for i, ele in enumerate(elem_type):
            proj[ele] = img_all[elem_ch_indices[i]] / proj_ic
            proj[ele] = proj[ele] / em_cs[ele] / rho[ele]
            proj[ele] = proj[ele] * pix ** 2
            proj[ele] = proj[ele] * rho[ele]
            proj[ele] = proj[ele] / M[ele]
            proj[ele] = proj[ele] * scale
            # X crop: crop_x_end >= 0 means an explicit end pixel; -1 = full width
            if crop_x_start > 0 or crop_x_end >= 0:
                x_end = crop_x_end if crop_x_end >= 0 else proj[ele].shape[2]
                proj[ele] = proj[ele][:, :, crop_x_start:x_end]
            # Y crop: crop_y_end >= 0 means an explicit end row; -1 = full height
            if crop_y_start > 0 or crop_y_end >= 0:
                y_end = crop_y_end if crop_y_end >= 0 else proj[ele].shape[1]
                proj[ele] = proj[ele][:, crop_y_start:y_end, :]
        x_crop_str = f"{crop_x_start}:{crop_x_end if crop_x_end >= 0 else 'end'}"
        y_crop_str = f"{crop_y_start}:{crop_y_end if crop_y_end >= 0 else 'end'}"
        _put_log(f"Projections normalized. X crop: [{x_crop_str}]  Y crop: [{y_crop_str}]")

        # ── Step 4: Bin projections ───────────────────────────────────────────
        _check_stop()
        current_step += 1
        _put_progress(current_step, total_steps, "Binning projections...")
        proj_raw = core.pre_treat([proj[ele] for ele in elem_type])
        s_pr = proj_raw.shape
        proj_bin = FL.bin_ndarray(
            proj_raw,
            (s_pr[0], s_pr[1], s_pr[2] // b, s_pr[3] // b),
            "sum",
        )
        _put_log(f"Projections binned: {proj_bin.shape}")

        # ── Steps 5+: Initial reconstruction per element ──────────────────────
        rec3D = {}
        for i, ele in enumerate(elem_type):
            _check_stop()
            current_step += 1
            _put_progress(
                current_step, total_steps,
                f"Reconstructing {ele} ({i + 1}/{n_elem})..."
            )
            rec3D[ele] = FL.recon_astra_sub(
                proj[ele], theta_tomopy, method=recon_method, num_iter=recon_n_iter
            )
            _put_log(f"  {ele} reconstruction done, shape={rec3D[ele].shape}")

        # ── Step: Save initial reconstructions ────────────────────────────────
        _check_stop()
        current_step += 1
        _put_progress(current_step, total_steps, "Saving initial reconstructions...")
        recon_raw = core.pre_treat([rec3D[ele] for ele in elem_type])
        FL.save_recon(fn_root, recon_raw, elem_type, -2)
        s_rr = recon_raw.shape
        recon_bin = FL.bin_ndarray(
            recon_raw,
            (s_rr[0], s_rr[1] // b, s_rr[2] // b, s_rr[3] // b),
            "sum",
        )
        FL.save_recon(fn_root, recon_bin, elem_type, -1)
        _put_log("Initial reconstructions saved (iter -2=full, -1=binned)")

        # ── Step: Generate / load detector mask ───────────────────────────────
        _check_stop()
        current_step += 1
        _put_progress(current_step, total_steps, "Preparing detector mask...")
        fn_mask = os.path.join(fn_root, f"mask3D_{mask_length}.h5")
        if os.path.exists(fn_mask):
            _put_log(f"Loading existing detector mask: {fn_mask}")
            mask3D = core.load_mask3D(fn_mask)
        else:
            _put_log(
                f"Computing detector mask "
                f"(alfa={det_alfa}°, theta={det_theta}°, length={mask_length})..."
            )
            mask3D = core.prep_detector_mask3D(
                alfa=det_alfa, theta=det_theta,
                length_maximum=mask_length, fn_save=fn_mask,
            )
            _put_log(f"Detector mask saved to {fn_mask}")
        core.load_global_mask(mask3D)

        # ── Iterative correction ──────────────────────────────────────────────
        recon_cor = recon_bin.copy()
        ref_prj   = proj_bin
        fpath_save = os.path.join(fn_root, "recon")
        os.makedirs(fpath_save, exist_ok=True)

        for it in range(1, n_corr_iters + 1):
            _check_stop()
            ts = time.time()

            current_step += 1
            _put_progress(
                current_step, total_steps,
                f"Iteration {it}/{n_corr_iters}: computing attenuation..."
            )

            recon_cor = core.smooth_filter(recon_cor, smooth_size)
            recon_cor = FL.rm_boarder(recon_cor, border_pixels)

            fsave_iter = os.path.join(fn_root, f"Angle_prj_{it:02d}")
            core.cal_and_save_atten_prj(
                param, cs, recon_cor, angle_list, ref_prj,
                fsave=fsave_iter, align_flag=False,
                enable_scale=False, num_cpu=num_cpu,
            )
            _put_log(f"  Attenuation computed in {time.time() - ts:.1f}s")

            for i, elem in enumerate(elem_type):
                _check_stop()
                current_step += 1
                _put_progress(
                    current_step, total_steps,
                    f"Iteration {it}/{n_corr_iters}: correcting {elem} ({i + 1}/{n_elem})..."
                )

                ref_tomo = np.ones(recon_cor[i].shape)

                if use_gpu:
                    cor = core.cuda_absorption_correction_wrap(
                        elem, ref_tomo, angle_list, fsave_iter,
                        corr_n_iter, True, fpath_save,
                    )
                else:
                    cor = core.absorption_correction_mpi(
                        elem, ref_tomo, angle_list, fsave_iter,
                        corr_n_iter, num_cpu, True, fpath_save,
                    )
                recon_cor[i] = FL.rm_boarder(cor, border_pixels)

            FL.save_recon(fn_root, recon_cor, elem_type, it)
            _put_log(
                f"Iteration {it} complete in {time.time() - ts:.1f}s, saved to {fpath_save}/"
            )

        # ── Done ──────────────────────────────────────────────────────────────
        _put_progress(total_steps, total_steps, "Correction complete!")
        _put_log(f"All results saved to {fpath_save}")
        status_queue.put({"recon_file": fpath_save})

    except RuntimeError as e:
        if "stopped by user" in str(e):
            _put_log("FL correction stopped by user.", level="WARNING")
        else:
            logger.exception("RuntimeError in fl_correction_worker_process")
            _put_log(str(e), level="ERROR")
            status_queue.put({"error": str(e)})
    except Exception as e:
        logger.exception("Error in fl_correction_worker_process")
        _put_log(f"Worker error: {e}", level="ERROR")
        status_queue.put({"error": str(e)})
    finally:
        status_queue.put({"finished": True})
