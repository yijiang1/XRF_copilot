"""Worker process that wraps simulate_XRF_maps for multiprocessing execution."""

import os
import sys
import time
import logging
import io
import contextlib


class StatusQueueHandler(logging.Handler):
    """Logging handler that sends log entries to a multiprocessing Queue."""

    def __init__(self, status_queue):
        super().__init__()
        self.status_queue = status_queue

    def emit(self, record):
        log_entry = self.format(record)
        self.status_queue.put(
            {
                "log": log_entry,
                "level": record.levelname,
                "timestamp": time.time(),
            }
        )


def setup_worker_logger(status_queue):
    """Set up the worker logger with console and queue handlers."""
    worker_logger = logging.getLogger("worker")
    worker_logger.setLevel(logging.INFO)
    worker_logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    worker_logger.addHandler(console_handler)

    queue_handler = StatusQueueHandler(status_queue)
    queue_handler.setFormatter(formatter)
    worker_logger.addHandler(queue_handler)

    return worker_logger


def simulation_worker_process(params, status_queue, stop_event):
    """Worker process that runs simulate_XRF_maps with progress reporting.

    Args:
        params: Dict of simulation parameters.
        status_queue: multiprocessing.Queue for sending status updates.
        stop_event: multiprocessing.Event for signaling stop.
    """
    try:
        worker_logger = setup_worker_logger(status_queue)
        worker_logger.info("XRF simulation worker process started")

        # Add the simulation package to path
        sim_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulation")
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from src.simulation.simulation import simulate_XRF_maps

        # Pre-compute total batches from ground truth file
        import numpy as np
        X = np.load(params["ground_truth_file"])
        sample_size_n = X.shape[1]
        sample_height_n = X.shape[3]
        batch_size = sample_height_n
        n_batch = (sample_height_n * sample_size_n) // batch_size
        del X

        status_queue.put({"total_batches": n_batch})
        worker_logger.info(f"Total batches: {n_batch}")

        # Progress callback that reports to the queue
        def progress_callback(current_batch, total_batches):
            if stop_event.is_set():
                raise InterruptedError("Simulation stopped by user")
            status_queue.put({
                "current_batch": current_batch,
                "total_batches": total_batches,
            })

        # Capture stdout for log relay
        worker_logger.info("Starting XRF simulation...")
        status_queue.put({"log": "Starting XRF simulation...", "level": "INFO", "timestamp": time.time()})

        sim_xrf_file, sim_xrt_file = simulate_XRF_maps(params, progress_callback=progress_callback)

        worker_logger.info(f"Simulation complete: {sim_xrf_file}")
        status_queue.put({
            "finished": True,
            "sim_xrf_file": sim_xrf_file,
            "sim_xrt_file": sim_xrt_file,
        })

    except InterruptedError:
        worker_logger.info("Simulation stopped by user")
        status_queue.put({"log": "Simulation stopped by user", "level": "WARNING", "timestamp": time.time()})
        status_queue.put({"finished": True})
    except Exception as e:
        print(f"Worker process error: {str(e)}")
        if "worker_logger" in locals():
            worker_logger.error(f"Error in worker process: {str(e)}")
        status_queue.put({"error": str(e)})
    finally:
        status_queue.put({"finished": True})
