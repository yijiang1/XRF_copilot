from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..recon_worker import reconstruction_worker_process
from ..queue_handlers import monitor_recon_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def recon_run_endpoint(recon_process_status, recon_output, recon_latest_worker_status, recon_worker_logs):
    """Register the run reconstruction endpoint."""

    @router.post("/run_reconstruction/")
    async def run_reconstruction():
        if recon_process_status["is_running"]:
            return {
                "status": "A reconstruction is already running.",
                "output": "Please wait for the current task to complete.",
            }

        try:
            # Reset output
            recon_output["error"] = None
            recon_output["current_epoch"] = 0
            recon_output["total_epochs"] = 0
            recon_output["recon_file"] = ""
            recon_worker_logs.clear()

            # Get stored params
            params = recon_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_reconstruction/ first.",
                )

            # Create communication channels
            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=reconstruction_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(f"Reconstruction worker process started with PID: {process.pid}")

            # Update process status
            recon_process_status["is_running"] = True
            recon_process_status["process"] = process
            recon_process_status["status_queue"] = status_queue
            recon_process_status["stop_event"] = stop_event

            # Start monitor thread
            monitor_thread = threading.Thread(
                target=monitor_recon_queue,
                args=(
                    recon_process_status,
                    recon_output,
                    recon_latest_worker_status,
                    recon_worker_logs,
                ),
                daemon=True,
            )
            monitor_thread.start()
            logger.info("Reconstruction monitor thread started")

            return {
                "status": "Reconstruction started in background.",
                "output": "Task submitted to run in background.",
            }

        except Exception as e:
            logger.error(f"Error starting reconstruction: {str(e)}")
            recon_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return run_reconstruction
