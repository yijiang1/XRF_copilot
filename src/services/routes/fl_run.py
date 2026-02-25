from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..fl_worker import fl_correction_worker_process
from ..queue_handlers import monitor_fl_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def fl_run_endpoint(fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs):
    """Register the run FL correction endpoint."""

    @router.post("/run_fl_correction/")
    async def run_fl_correction():
        if fl_process_status["is_running"]:
            return {
                "status": "FL correction is already running.",
                "output": "Please wait for the current task to complete.",
            }

        try:
            # Reset output state
            fl_output["error"] = None
            fl_output["current_step"] = 0
            fl_output["total_steps"] = 0
            fl_output["step_label"] = ""
            fl_output["recon_file"] = ""
            fl_worker_logs.clear()

            params = fl_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_fl_correction/ first.",
                )

            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=fl_correction_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(f"FL correction worker process started with PID: {process.pid}")

            fl_process_status["is_running"] = True
            fl_process_status["process"] = process
            fl_process_status["status_queue"] = status_queue
            fl_process_status["stop_event"] = stop_event

            monitor_thread = threading.Thread(
                target=monitor_fl_queue,
                args=(fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs),
                daemon=True,
            )
            monitor_thread.start()
            logger.info("FL correction monitor thread started")

            return {
                "status": "FL correction started in background.",
                "output": "Task submitted to run in background.",
            }

        except Exception as e:
            logger.error(f"Error starting FL correction: {str(e)}")
            fl_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return run_fl_correction
