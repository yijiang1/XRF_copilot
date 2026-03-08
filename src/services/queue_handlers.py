"""Background thread to monitor the worker process status queue."""

import time
import logging
import queue as queue_module

MAX_WORKER_LOGS = 1000  # cap per-session log buffer

logger = logging.getLogger(__name__)


def monitor_process_queue(
    process_status, simulation_output, latest_worker_status, worker_logs
):
    """Background thread that reads from the worker's status queue.

    Args:
        process_status: Dict with is_running, process, status_queue, stop_event.
        simulation_output: Dict to store simulation results and progress.
        latest_worker_status: Dict with timestamp and status.
        worker_logs: List to accumulate worker log entries.
    """
    logger.info("Monitor thread started")

    while process_status["is_running"]:
        try:
            alive = process_status["process"].is_alive()

            try:
                status = process_status["status_queue"].get(
                    block=alive, timeout=0.5 if alive else 0,
                )

                if "error" in status:
                    logger.error(f"Received error from worker: {status['error']}")
                    simulation_output["error"] = status["error"]
                    process_status["is_running"] = False
                    break

                if "current_batch" in status:
                    simulation_output["current_batch"] = status["current_batch"]

                if "total_batches" in status:
                    simulation_output["total_batches"] = status["total_batches"]

                if "log" in status:
                    worker_logs.append(
                        {
                            "message": status["log"],
                            "level": status.get("level", "WORKER"),
                            "timestamp": status.get("timestamp", time.time()),
                        }
                    )

                if "sim_xrf_file" in status:
                    simulation_output["sim_xrf_file"] = status["sim_xrf_file"]
                if "sim_xrt_file" in status:
                    simulation_output["sim_xrt_file"] = status["sim_xrt_file"]

                if "finished" in status and status["finished"]:
                    logger.info("Worker process finished")
                    process_status["is_running"] = False
                    break

            except queue_module.Empty:
                if not alive:
                    logger.info("Worker process exited, queue drained")
                    process_status["is_running"] = False
                    break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in monitor thread: {str(e)}")
            simulation_output["error"] = f"Error in monitor thread: {str(e)}"
            process_status["is_running"] = False
            break

    # Clean up
    logger.info("Cleaning up resources in monitor thread")
    if process_status["process"] and process_status["process"].is_alive():
        if process_status["stop_event"]:
            process_status["stop_event"].set()
        time.sleep(1)
        if process_status["process"].is_alive():
            process_status["process"].kill()
            process_status["process"].join(timeout=5)

    # Drain queue
    if process_status["status_queue"]:
        try:
            while True:
                process_status["status_queue"].get(block=False)
        except:
            pass

    # Reset state
    simulation_output["current_batch"] = 0
    simulation_output["total_batches"] = 0
    latest_worker_status["timestamp"] = 0
    latest_worker_status["status"] = None
    process_status["process"] = None
    process_status["status_queue"] = None
    process_status["stop_event"] = None
    logger.info("Monitor thread finished")


def monitor_recon_queue(
    recon_process_status, recon_output, recon_latest_worker_status, recon_worker_logs
):
    """Background thread that reads from the reconstruction worker's status queue.

    Args:
        recon_process_status: Dict with is_running, process, status_queue, stop_event.
        recon_output: Dict to store reconstruction results and progress.
        recon_latest_worker_status: Dict with timestamp and status.
        recon_worker_logs: List to accumulate worker log entries.
    """
    logger.info("Reconstruction monitor thread started")

    while recon_process_status["is_running"]:
        try:
            alive = recon_process_status["process"].is_alive()

            try:
                status = recon_process_status["status_queue"].get(
                    block=alive, timeout=0.5 if alive else 0,
                )

                if "error" in status:
                    logger.error(f"Received error from recon worker: {status['error']}")
                    recon_output["error"] = status["error"]
                    recon_process_status["is_running"] = False
                    break

                if "current_epoch" in status:
                    recon_output["current_epoch"] = status["current_epoch"]

                if "total_epochs" in status:
                    recon_output["total_epochs"] = status["total_epochs"]

                if "log" in status:
                    recon_worker_logs.append(
                        {
                            "message": status["log"],
                            "level": status.get("level", "WORKER"),
                            "timestamp": status.get("timestamp", time.time()),
                        }
                    )

                if "recon_file" in status:
                    recon_output["recon_file"] = status["recon_file"]

                if "finished" in status and status["finished"]:
                    logger.info("Reconstruction worker process finished")
                    recon_process_status["is_running"] = False
                    break

            except queue_module.Empty:
                if not alive:
                    logger.info("Reconstruction worker exited, queue drained")
                    recon_process_status["is_running"] = False
                    break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in reconstruction monitor thread: {str(e)}")
            recon_output["error"] = f"Error in monitor thread: {str(e)}"
            recon_process_status["is_running"] = False
            break

    # Clean up
    logger.info("Cleaning up reconstruction resources in monitor thread")
    if recon_process_status["process"] and recon_process_status["process"].is_alive():
        if recon_process_status["stop_event"]:
            recon_process_status["stop_event"].set()
        time.sleep(1)
        if recon_process_status["process"].is_alive():
            recon_process_status["process"].kill()
            recon_process_status["process"].join(timeout=5)

    # Drain queue
    if recon_process_status["status_queue"]:
        try:
            while True:
                recon_process_status["status_queue"].get(block=False)
        except:
            pass

    # Reset state
    recon_output["current_epoch"] = 0
    recon_output["total_epochs"] = 0
    recon_latest_worker_status["timestamp"] = 0
    recon_latest_worker_status["status"] = None
    recon_process_status["process"] = None
    recon_process_status["status_queue"] = None
    recon_process_status["stop_event"] = None
    logger.info("Reconstruction monitor thread finished")


def monitor_fl_queue(
    fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs
):
    """Background thread that reads from the FL correction worker's status queue."""
    logger.info("FL correction monitor thread started")

    while fl_process_status["is_running"]:
        try:
            alive = fl_process_status["process"].is_alive()

            try:
                status = fl_process_status["status_queue"].get(
                    block=alive, timeout=0.5 if alive else 0,
                )

                if "error" in status:
                    logger.error(f"Received error from FL worker: {status['error']}")
                    fl_output["error"] = status["error"]
                    fl_process_status["is_running"] = False
                    break

                if "current_step" in status:
                    fl_output["current_step"] = status["current_step"]

                if "total_steps" in status:
                    fl_output["total_steps"] = status["total_steps"]

                if "step_label" in status:
                    fl_output["step_label"] = status["step_label"]

                if "log" in status:
                    fl_worker_logs.append({
                        "message": status["log"],
                        "level": status.get("level", "WORKER"),
                        "timestamp": status.get("timestamp", time.time()),
                    })

                if "recon_file" in status:
                    fl_output["recon_file"] = status["recon_file"]

                if "finished" in status and status["finished"]:
                    logger.info("FL correction worker finished")
                    fl_process_status["is_running"] = False
                    break

            except queue_module.Empty:
                if not alive:
                    logger.info("FL correction worker exited, queue drained")
                    fl_process_status["is_running"] = False
                    break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in FL correction monitor thread: {str(e)}")
            fl_output["error"] = f"Error in monitor thread: {str(e)}"
            fl_process_status["is_running"] = False
            break

    # Clean up
    logger.info("Cleaning up FL correction resources in monitor thread")
    if fl_process_status["process"] and fl_process_status["process"].is_alive():
        if fl_process_status["stop_event"]:
            fl_process_status["stop_event"].set()
        time.sleep(1)
        if fl_process_status["process"].is_alive():
            fl_process_status["process"].kill()
            fl_process_status["process"].join(timeout=5)

    # Drain queue
    if fl_process_status["status_queue"]:
        try:
            while True:
                fl_process_status["status_queue"].get(block=False)
        except:
            pass

    # Reset state
    fl_output["current_step"] = 0
    fl_output["total_steps"] = 0
    fl_output["step_label"] = ""
    fl_latest_worker_status["timestamp"] = 0
    fl_latest_worker_status["status"] = None
    fl_process_status["process"] = None
    fl_process_status["status_queue"] = None
    fl_process_status["stop_event"] = None
    logger.info("FL correction monitor thread finished")


def monitor_session_queue(session):
    """Generic background monitor thread for any XRFSession.

    Reads from ``session.process_status["status_queue"]`` and writes progress /
    logs into ``session.output`` and ``session.worker_logs``.  Works for BNL,
    Panpan, and Wendy sessions.

    Args:
        session: An ``XRFSession`` instance (from session_manager.py).
    """
    ps = session.process_status
    out = session.output
    logs = session.worker_logs
    method = session.method
    logger.info(f"Session monitor started for {session.session_id} ({session.display_name})")

    while ps["is_running"]:
        try:
            alive = ps["process"].is_alive()

            try:
                status = ps["status_queue"].get(
                    block=alive, timeout=0.5 if alive else 0,
                )

                if "error" in status:
                    logger.error(f"Worker error [{session.session_id}]: {status['error']}")
                    out["error"] = status["error"]
                    ps["is_running"] = False
                    break

                # BNL progress fields
                if "current_step" in status:
                    out["current_step"] = status["current_step"]
                if "total_steps" in status:
                    out["total_steps"] = status["total_steps"]
                if "step_label" in status:
                    out["step_label"] = status["step_label"]

                # Panpan / Wendy progress fields
                if "current_epoch" in status:
                    out["current_epoch"] = status["current_epoch"]
                if "total_epochs" in status:
                    out["total_epochs"] = status["total_epochs"]

                # Log entry
                if "log" in status:
                    entry = {
                        "message": status["log"],
                        "level": status.get("level", "WORKER"),
                        "timestamp": status.get("timestamp", time.time()),
                    }
                    logs.append(entry)
                    if len(logs) > MAX_WORKER_LOGS:
                        del logs[: len(logs) - MAX_WORKER_LOGS]

                # Output path (announced at worker start)
                if "recon_path" in status:
                    out["recon_path"] = status["recon_path"]

                # Result file
                if "recon_file" in status:
                    out["recon_file"] = status["recon_file"]

                # Checkpoint saved — store for result viewer
                if "checkpoint_saved" in status:
                    out["latest_checkpoint"] = {
                        "file": status.get("checkpoint_file", ""),
                        "iteration": status.get("iteration", -1),
                    }

                if status.get("finished"):
                    logger.info(f"Worker finished for session {session.session_id}")
                    ps["is_running"] = False
                    break

            except queue_module.Empty:
                if not alive:
                    logger.info(f"Worker exited for session {session.session_id}, queue drained")
                    ps["is_running"] = False
                    break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Monitor error [{session.session_id}]: {e}")
            out["error"] = f"Monitor error: {e}"
            ps["is_running"] = False
            break

    # ── Clean up ──────────────────────────────────────────────────────────────
    if ps["process"] and ps["process"].is_alive():
        if ps["stop_event"]:
            ps["stop_event"].set()
        time.sleep(1)
        if ps["process"].is_alive():
            ps["process"].kill()
            ps["process"].join(timeout=5)

    if ps["status_queue"]:
        try:
            while True:
                ps["status_queue"].get(block=False)
        except Exception:
            pass

    session.latest_worker_status["timestamp"] = 0
    session.latest_worker_status["status"] = None
    ps["process"] = None
    ps["status_queue"] = None
    ps["stop_event"] = None
    logger.info(f"Session monitor finished for {session.session_id}")
