from fastapi import APIRouter
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()


def stop_endpoint(process_status, simulation_output, latest_worker_status):
    """Register the stop simulation endpoint."""

    @router.post("/stop_simulation/")
    async def stop_simulation():
        if not process_status["is_running"]:
            return {"status": "No simulation is running."}

        try:
            logger.info("Stopping simulation process")

            if process_status["stop_event"]:
                process_status["stop_event"].set()

            time.sleep(0.5)

            if process_status["process"] and process_status["process"].is_alive():
                process_status["process"].kill()
                process_status["process"].join(timeout=5)

            # Drain queue
            if process_status["status_queue"]:
                try:
                    while True:
                        process_status["status_queue"].get(block=False)
                except Exception:
                    pass

            # Reset state
            process_status["is_running"] = False
            process_status["process"] = None
            process_status["status_queue"] = None
            process_status["stop_event"] = None

            simulation_output["error"] = None
            simulation_output["current_batch"] = 0
            simulation_output["total_batches"] = 0
            latest_worker_status["timestamp"] = 0
            latest_worker_status["status"] = None

            return {"status": "Simulation stopped successfully."}
        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")
            return {"status": "Error stopping simulation", "error": str(e)}

    return stop_simulation
