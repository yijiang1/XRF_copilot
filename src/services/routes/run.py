from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..worker import simulation_worker_process
from ..queue_handlers import monitor_process_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def run_endpoint(process_status, simulation_output, latest_worker_status, worker_logs):
    """Register the run simulation endpoint."""

    @router.post("/run_simulation/")
    async def run_simulation():
        if process_status["is_running"]:
            return {
                "status": "A simulation is already running.",
                "output": "Please wait for the current task to complete.",
            }

        try:
            # Reset output
            simulation_output["error"] = None
            simulation_output["current_batch"] = 0
            simulation_output["total_batches"] = 0
            simulation_output["sim_xrf_file"] = ""
            simulation_output["sim_xrt_file"] = ""
            worker_logs.clear()

            # Get stored params
            params = simulation_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_simulation/ first.",
                )

            # Create communication channels
            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=simulation_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(f"Worker process started with PID: {process.pid}")

            # Update process status
            process_status["is_running"] = True
            process_status["process"] = process
            process_status["status_queue"] = status_queue
            process_status["stop_event"] = stop_event

            # Start monitor thread
            monitor_thread = threading.Thread(
                target=monitor_process_queue,
                args=(
                    process_status,
                    simulation_output,
                    latest_worker_status,
                    worker_logs,
                ),
                daemon=True,
            )
            monitor_thread.start()
            logger.info("Monitor thread started")

            return {
                "status": "Simulation started in background.",
                "output": "Task submitted to run in background.",
            }

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
            simulation_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return run_simulation
