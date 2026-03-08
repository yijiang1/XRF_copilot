"""Worker process that wraps reconstruct_di_xrftomo for multiprocessing execution."""

import os
import sys
import time
import logging
import numpy as np


class StatusQueueHandler(logging.Handler):
    """Logging handler that sends log entries to a multiprocessing Queue."""

    def __init__(self, status_queue):
        super().__init__()
        self.status_queue = status_queue

    def emit(self, record):
        self.status_queue.put({
            "log": self.format(record),
            "level": record.levelname,
            "timestamp": time.time(),
        })


def setup_worker_logger(status_queue):
    logger = logging.getLogger("di_worker")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    qh = StatusQueueHandler(status_queue)
    qh.setFormatter(fmt)
    logger.addHandler(qh)
    return logger


def _detect_elements_from_channels(channel_names):
    """Auto-detect elements and fluorescence lines from HDF5 channel names.

    Rules:
      'Ca'   → ('Ca', 'K')   bare name → K-shell
      'Ca_L' → ('Ca', 'L')   _L suffix → L-shell
      'Ca_M' → ('Ca', 'M')   _M suffix → M-shell
      'us_ic', 'abs_ic', etc. → skipped (base not a valid xraylib element)

    Returns:
      this_aN_dic               dict  e.g. {'Ca': 20, 'Sc': 21}
      element_lines_roi         ndarray shape (n_lines, 2)
      n_line_group_each_element ndarray shape (n_elements,)
    """
    import xraylib as xlib
    element_lines = []
    for raw in channel_names:
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        parts = name.split("_")
        base = parts[0]
        try:
            xlib.SymbolToAtomicNumber(base)
        except Exception:
            continue
        shell = parts[1] if len(parts) > 1 and parts[1] in ("K", "L", "M") else "K"
        element_lines.append((base, shell))

    seen = set()
    unique_elements = []
    for sym, _ in element_lines:
        if sym not in seen:
            seen.add(sym)
            unique_elements.append(sym)

    this_aN_dic = {sym: xlib.SymbolToAtomicNumber(sym) for sym in unique_elements}
    element_lines_roi = np.array([[sym, shell] for sym, shell in element_lines])
    n_line_group_each_element = np.array([
        sum(1 for sym, _ in element_lines if sym == s) for s in unique_elements
    ])
    return this_aN_dic, element_lines_roi, n_line_group_each_element


def di_reconstruction_worker_process(params: dict, status_queue, stop_event):
    """Worker process that runs reconstruct_di_xrftomo with progress reporting.

    Args:
        params:       Dict matching DiReconParams fields.
        status_queue: multiprocessing.Queue for status updates.
        stop_event:   multiprocessing.Event for graceful stop.
    """
    try:
        worker_logger = setup_worker_logger(status_queue)
        worker_logger.info("Di et al. reconstruction worker process started")

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        import xraylib as xlib
        import torch as tc
        from src.di_recon.di_reconstruction import reconstruct_di_xrftomo

        # ── Set up device ──
        gpu_id = params.get("gpu_id", 3)
        if tc.cuda.is_available():
            dev = tc.device(f"cuda:{gpu_id}")
        else:
            dev = tc.device("cpu")
        worker_logger.info(f"Using device: {dev}")

        # ── Build FL line arrays ──
        fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE,
                         xlib.KB1_LINE, xlib.KB2_LINE, xlib.KB3_LINE,
                         xlib.KB4_LINE, xlib.KB5_LINE])
        fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE,
                         xlib.LB2_LINE, xlib.LB3_LINE, xlib.LB4_LINE,
                         xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE,
                         xlib.LB9_LINE, xlib.LB10_LINE, xlib.LB15_LINE,
                         xlib.LB17_LINE])
        fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

        n_outer_epochs = params["n_outer_epochs"]
        status_queue.put({"total_epochs": n_outer_epochs})
        worker_logger.info(f"Total outer epochs: {n_outer_epochs}")

        # ── Progress + stop callback ──
        _last_epoch_time = {"t": time.time()}

        def progress_callback(current_epoch, total_epochs):
            if stop_event.is_set():
                raise InterruptedError("Di reconstruction stopped by user")
            now = time.time()
            elapsed = now - _last_epoch_time["t"]
            _last_epoch_time["t"] = now
            status_queue.put({
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
            })
            status_queue.put({
                "log": f"[Di] Epoch {current_epoch}/{total_epochs} complete ({elapsed:.1f}s)",
                "level": "INFO",
                "timestamp": now,
            })

        def log_callback(msg):
            status_queue.put({"log": msg, "level": "INFO", "timestamp": time.time()})

        status_queue.put({
            "log": "Starting Di et al. reconstruction...",
            "level": "INFO",
            "timestamp": time.time(),
        })

        # ── Auto-detect grid dims and elements from HDF5 ──
        import h5py
        data_file = os.path.join(params["fn_root"], params["fn_data"])
        with h5py.File(data_file, "r") as _f:
            _shape = _f["data"].shape          # (n_ch, n_ang, H, W)
            _channel_names = _f["elements"][:] # bytes array of channel names
        this_aN_dic, element_lines_roi, n_line_group_each_element = \
            _detect_elements_from_channels(_channel_names)
        worker_logger.info(
            f"Auto-detected elements: {list(this_aN_dic.keys())}, "
            f"lines: {element_lines_roi.tolist()}"
        )

        # ── Filter elements to user selection and build emission energy override ──
        fl_energy_override = None
        elem_symbols_str    = params.get("element_symbols", "").strip()
        emission_energy_str = params.get("emission_energy", "").strip()
        if elem_symbols_str and len(element_lines_roi) > 0:
            selected_ch_names = [x.strip() for x in elem_symbols_str.split(",") if x.strip()]
            selected_lines = set()
            for ch_name in selected_ch_names:
                parts = ch_name.split("_")
                base  = parts[0]
                shell = parts[1] if len(parts) > 1 and parts[1] in ("K", "L", "M") else "K"
                selected_lines.add((base, shell))
            mask = np.array([(row[0], row[1]) in selected_lines for row in element_lines_roi])
            filtered = element_lines_roi[mask]
            if len(filtered) > 0:
                element_lines_roi = filtered
                seen, unique_elems = set(), []
                for sym, _ in element_lines_roi:
                    if sym not in seen:
                        seen.add(sym); unique_elems.append(sym)
                this_aN_dic = {sym: xlib.SymbolToAtomicNumber(sym) for sym in unique_elems}
                n_line_group_each_element = np.array([
                    sum(1 for sym, _ in element_lines_roi if sym == s) for s in unique_elems
                ])
                worker_logger.info(
                    f"User-selected elements: {list(this_aN_dic.keys())}, "
                    f"lines: {element_lines_roi.tolist()}"
                )
            if emission_energy_str:
                em_energies = [float(x.strip()) for x in emission_energy_str.split(",") if x.strip()]
                if len(selected_ch_names) == len(em_energies):
                    fl_energy_override = {}
                    for ch_name, em_e in zip(selected_ch_names, em_energies):
                        parts = ch_name.split("_")
                        base  = parts[0]
                        shell = parts[1] if len(parts) > 1 and parts[1] in ("K", "L", "M") else "K"
                        if em_e > 0:
                            fl_energy_override[(base, shell)] = em_e
                    worker_logger.info(f"Emission energy override: {fl_energy_override}")

        sample_height_n = int(_shape[2])
        sample_size_n   = int(_shape[3])
        pixel_size_nm   = float(params["pixel_size_nm"])
        sample_size_cm  = sample_size_n * pixel_size_nm * 1e-7
        worker_logger.info(
            f"Data shape: {_shape} → sample_size_n={sample_size_n}, "
            f"sample_height_n={sample_height_n}, "
            f"sample_size_cm={sample_size_cm:.6f} cm (pixel={pixel_size_nm} nm)"
        )

        # ── Output directory ──
        recon_path = os.path.join(params["fn_root"], "Wendy")
        os.makedirs(recon_path, exist_ok=True)
        # Announce output path so result viewer can scan for checkpoints
        status_queue.put({"recon_path": recon_path})

        # ── Auto-generate output filenames (unified across methods) ──
        _date = time.strftime("%Y%m%d")
        f_recon_parameters = f"recon_parameters_{_date}.txt"
        f_recon_grid       = "recon"
        f_initial_guess    = "recon_initial"

        # ── Auto-generate P matrix path ──
        P_folder = os.path.join(recon_path, "P_array")
        os.makedirs(P_folder, exist_ok=True)
        f_P = (
            f"Intersecting_Length_{sample_size_n}x{sample_height_n}"
            f"_pix{pixel_size_nm:.0f}nm"
            f"_dia{params['det_dia_cm']:.4g}cm"
            f"_dist{params['det_from_sample_cm']:.4g}cm"
            f"_spc{params['det_ds_spacing_cm']:.4g}cm"
            f"_{params['det_on_which_side']}"
        )

        recon_file = reconstruct_di_xrftomo(
            dev=dev,
            data_path=params["fn_root"],
            f_XRF_data=params["fn_data"],
            f_XRT_data=params["fn_data"],   # single file: exchange/data contains all channels
            recon_path=recon_path,
            P_folder=P_folder,
            f_P=f_P,
            f_recon_grid=f_recon_grid,
            f_initial_guess=f_initial_guess,
            f_recon_parameters=f_recon_parameters,
            sample_size_n=sample_size_n,
            sample_height_n=sample_height_n,
            sample_size_cm=sample_size_cm,
            this_aN_dic=this_aN_dic,
            element_lines_roi=element_lines_roi,
            n_line_group_each_element=n_line_group_each_element,
            probe_energy=np.array([params["probe_energy"]]),
            probe_intensity=params["probe_intensity"],
            probe_att=params["probe_att"],
            manual_det_coord=params["manual_det_coord"],
            set_det_coord_cm=None,
            det_on_which_side=params["det_on_which_side"],
            manual_det_area=params["manual_det_area"],
            det_area_cm2=None,
            det_dia_cm=params["det_dia_cm"],
            det_ds_spacing_cm=params["det_ds_spacing_cm"],
            det_from_sample_cm=params["det_from_sample_cm"],
            XRT_ratio_dataset_idx=params["XRT_ratio_dataset_idx"],
            scaler_counts_us_ic_dataset_idx=params["scaler_counts_us_ic_dataset_idx"],
            scaler_counts_ds_ic_dataset_idx=params["scaler_counts_ds_ic_dataset_idx"],
            theta_ls_dataset=params["theta_ls_dataset"],
            channel_names=params["channel_names"],
            loss_type=params["loss_type"],
            beta1_xrt=params["beta1_xrt"],
            tikhonov_lambda=params["tikhonov_lambda"],
            n_outer_epochs=n_outer_epochs,
            lbfgs_n_iter=params["lbfgs_n_iter"],
            lbfgs_history=params["lbfgs_history"],
            selfAb=params["selfAb"],
            ini_kind=params["ini_kind"],
            init_const=params["init_const"],
            cont_from_check_point=params["cont_from_check_point"],
            use_saved_initial_guess=params["use_saved_initial_guess"],
            minibatch_size=params["minibatch_size"],
            save_every_n_epochs=params["save_every_n_epochs"],
            fl_energy_override=fl_energy_override,
            progress_callback=progress_callback,
            log_callback=log_callback,
            fl_K=fl_K, fl_L=fl_L, fl_M=fl_M,
        )

        worker_logger.info(f"Di reconstruction complete: {recon_file}")
        status_queue.put({
            "finished": True,
            "recon_file": recon_file,
        })

    except InterruptedError:
        status_queue.put({
            "log": "Di reconstruction stopped by user",
            "level": "WARNING",
            "timestamp": time.time(),
        })
        status_queue.put({"finished": True})
    except Exception as e:
        import traceback
        print(f"Di reconstruction worker error: {e}")
        traceback.print_exc()
        if "worker_logger" in locals():
            worker_logger.error(f"Error in Di reconstruction worker: {e}")
        status_queue.put({"error": str(e)})
    finally:
        status_queue.put({"finished": True})
