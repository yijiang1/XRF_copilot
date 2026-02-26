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


def _parse_element_symbols(s: str) -> dict:
    """Parse 'Ca, Sc' → {'Ca': 20, 'Sc': 21} via xraylib atomic number lookup."""
    import xraylib
    result = {}
    for part in s.split(","):
        sym = part.strip()
        if sym:
            result[sym] = xraylib.SymbolToAtomicNumber(sym)
    return result


def _parse_element_lines_roi(s: str) -> np.ndarray:
    """Parse 'Ca K, Ca L, Sc K' → [['Ca','K'],['Ca','L'],['Sc','K']]."""
    rows = []
    for part in s.split(","):
        part = part.strip()
        tokens = part.split()
        if len(tokens) == 2:
            rows.append([tokens[0], tokens[1]])
    return np.array(rows)


def _parse_int_list(s: str) -> np.ndarray:
    """Parse '2, 2' → np.array([2, 2])."""
    return np.array([int(x.strip()) for x in s.split(",") if x.strip()])


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

        # ── Parse user-friendly string params ──
        this_aN_dic = _parse_element_symbols(params["element_symbols"])
        element_lines_roi = _parse_element_lines_roi(params["element_lines_roi_str"])
        n_line_group_each_element = _parse_int_list(params["n_line_group_each_element_str"])

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
        def progress_callback(current_epoch, total_epochs):
            if stop_event.is_set():
                raise InterruptedError("Di reconstruction stopped by user")
            status_queue.put({
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
            })

        status_queue.put({
            "log": "Starting Di et al. reconstruction...",
            "level": "INFO",
            "timestamp": time.time(),
        })

        recon_file = reconstruct_di_xrftomo(
            dev=dev,
            data_path=params["data_path"],
            f_XRF_data=params["f_XRF_data"],
            f_XRT_data=params["f_XRT_data"],
            recon_path=params["recon_path"],
            P_folder=params["P_folder"],
            f_P=params["f_P"],
            f_recon_grid=params["f_recon_grid"],
            f_initial_guess=params["f_initial_guess"],
            f_recon_parameters=params["f_recon_parameters"],
            sample_size_n=params["sample_size_n"],
            sample_height_n=params["sample_height_n"],
            sample_size_cm=params["sample_size_cm"],
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
            progress_callback=progress_callback,
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
