from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..di_worker import di_reconstruction_worker_process
from ..queue_handlers import monitor_recon_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def di_run_endpoint(di_process_status, di_output, di_latest_worker_status, di_worker_logs):
    """Register the run Di reconstruction endpoint."""

    @router.post("/run_di_reconstruction/")
    async def run_di_reconstruction():
        if di_process_status["is_running"]:
            return {
                "status": "A Di reconstruction is already running.",
                "output": "Please wait for the current task to complete.",
            }

        try:
            di_output["error"] = None
            di_output["current_epoch"] = 0
            di_output["total_epochs"] = 0
            di_output["recon_file"] = ""
            di_worker_logs.clear()

            params = di_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_di_reconstruction/ first.",
                )

            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=di_reconstruction_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(f"Di reconstruction worker started with PID: {process.pid}")

            di_process_status["is_running"] = True
            di_process_status["process"] = process
            di_process_status["status_queue"] = status_queue
            di_process_status["stop_event"] = stop_event

            monitor_thread = threading.Thread(
                target=monitor_recon_queue,
                args=(di_process_status, di_output, di_latest_worker_status, di_worker_logs),
                daemon=True,
            )
            monitor_thread.start()
            logger.info("Di reconstruction monitor thread started")

            return {
                "status": "Di reconstruction started in background.",
                "output": "Task submitted to run in background.",
            }

        except Exception as e:
            logger.error(f"Error starting Di reconstruction: {str(e)}")
            di_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return run_di_reconstruction
