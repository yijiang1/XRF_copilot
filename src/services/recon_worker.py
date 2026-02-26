"""Worker process that wraps reconstruct_jXRFT_tomography for multiprocessing execution."""

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
    logger = logging.getLogger("recon_worker")
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


def reconstruction_worker_process(params: dict, status_queue, stop_event):
    """Worker process that runs reconstruct_jXRFT_tomography with progress reporting.

    Args:
        params: Dict matching XRFReconstructionParams fields.
        status_queue: multiprocessing.Queue for status updates.
        stop_event: multiprocessing.Event for graceful stop.
    """
    try:
        worker_logger = setup_worker_logger(status_queue)
        worker_logger.info("XRF reconstruction worker process started")

        # Add project root to path so src.* imports work
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        import xraylib as xlib
        import torch as tc
        from src.reconstruction.XRF_tomography import reconstruct_jXRFT_tomography

        # ── Set up device ──
        gpu_id = params.get("gpu_id", 3)
        if tc.cuda.is_available():
            dev = tc.device(f"cuda:{gpu_id}")
        else:
            dev = tc.device("cpu")
        worker_logger.info(f"Using device: {dev}")

        # ── Parse user-friendly string params into reconstruction format ──
        this_aN_dic = _parse_element_symbols(params["element_symbols"])
        element_lines_roi = _parse_element_lines_roi(params["element_lines_roi_str"])
        n_line_group_each_element = _parse_int_list(params["n_line_group_each_element_str"])

        # ── Build fl_K, fl_L, fl_M lookup arrays ──
        fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE,
                         xlib.KB1_LINE, xlib.KB2_LINE, xlib.KB3_LINE,
                         xlib.KB4_LINE, xlib.KB5_LINE])
        fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE,
                         xlib.LB2_LINE, xlib.LB3_LINE, xlib.LB4_LINE,
                         xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE,
                         xlib.LB9_LINE, xlib.LB10_LINE, xlib.LB15_LINE,
                         xlib.LB17_LINE])
        fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

        n_epochs = params["n_epochs"]
        status_queue.put({"total_epochs": n_epochs})
        worker_logger.info(f"Total epochs: {n_epochs}")

        # ── Progress callback ──
        def progress_callback(current_epoch, total_epochs):
            if stop_event.is_set():
                raise InterruptedError("Reconstruction stopped by user")
            status_queue.put({
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
            })

        status_queue.put({"log": "Starting XRF reconstruction...", "level": "INFO", "timestamp": time.time()})

        # ── Build reconstruction kwargs ──
        recon_kwargs = dict(
            dev=dev,
            sample_size_n=params["sample_size_n"],
            sample_height_n=params["sample_height_n"],
            sample_size_cm=params["sample_size_cm"],
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
            use_std_calibation=False,
            std_path=None, f_std=None, std_element_lines_roi=None,
            density_std_elements=None, fitting_method=None,
            n_epochs=n_epochs,
            save_every_n_epochs=params["save_every_n_epochs"],
            minibatch_size=params["minibatch_size"],
            f_recon_parameters=params["f_recon_parameters"],
            selfAb=params["selfAb"],
            cont_from_check_point=params["cont_from_check_point"],
            use_saved_initial_guess=params["use_saved_initial_guess"],
            ini_kind=params["ini_kind"],
            init_const=params["init_const"],
            ini_rand_amp=0.1,
            recon_path=params["recon_path"],
            f_initial_guess=params["f_initial_guess"],
            f_recon_grid=params["f_recon_grid"],
            data_path=params["data_path"],
            f_XRF_data=params["f_XRF_data"],
            f_XRT_data=params["f_XRT_data"],
            scaler_counts_us_ic_dataset_idx=params["scaler_counts_us_ic_dataset_idx"],
            scaler_counts_ds_ic_dataset_idx=params["scaler_counts_ds_ic_dataset_idx"],
            XRT_ratio_dataset_idx=params["XRT_ratio_dataset_idx"],
            theta_ls_dataset=params["theta_ls_dataset"],
            channel_names=params["channel_names"],
            this_aN_dic=this_aN_dic,
            element_lines_roi=element_lines_roi,
            n_line_group_each_element=n_line_group_each_element,
            b1=params["b1"],
            b2=params["b2"],
            lr=params["lr"],
            P_folder=params["P_folder"],
            f_P=params["f_P"],
            fl_K=fl_K,
            fl_L=fl_L,
            fl_M=fl_M,
            progress_callback=progress_callback,
        )

        reconstruct_jXRFT_tomography(**recon_kwargs)

        # Final result file
        import os as _os
        recon_file = _os.path.join(params["recon_path"], params["f_recon_grid"] + ".h5")
        worker_logger.info(f"Reconstruction complete: {recon_file}")
        status_queue.put({
            "finished": True,
            "recon_file": recon_file,
        })

    except InterruptedError:
        status_queue.put({"log": "Reconstruction stopped by user", "level": "WARNING", "timestamp": time.time()})
        status_queue.put({"finished": True})
    except Exception as e:
        print(f"Reconstruction worker error: {e}")
        if "worker_logger" in locals():
            worker_logger.error(f"Error in reconstruction worker: {e}")
        status_queue.put({"error": str(e)})
    finally:
        status_queue.put({"finished": True})
