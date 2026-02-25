from fastapi import APIRouter
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()


def recon_stop_endpoint(recon_process_status, recon_output, recon_latest_worker_status):
    """Register the stop reconstruction endpoint."""

    @router.post("/stop_reconstruction/")
    async def stop_reconstruction():
        if not recon_process_status["is_running"]:
            return {"status": "No reconstruction is running."}

        try:
            logger.info("Stopping reconstruction process")

            if recon_process_status["stop_event"]:
                recon_process_status["stop_event"].set()

            time.sleep(0.5)

            if recon_process_status["process"] and recon_process_status["process"].is_alive():
                recon_process_status["process"].kill()
                recon_process_status["process"].join(timeout=5)

            # Drain queue
            if recon_process_status["status_queue"]:
                try:
                    while True:
                        recon_process_status["status_queue"].get(block=False)
                except Exception:
                    pass

            # Reset state
            recon_process_status["is_running"] = False
            recon_process_status["process"] = None
            recon_process_status["status_queue"] = None
            recon_process_status["stop_event"] = None

            recon_output["error"] = None
            recon_output["current_epoch"] = 0
            recon_output["total_epochs"] = 0
            recon_latest_worker_status["timestamp"] = 0
            recon_latest_worker_status["status"] = None

            return {"status": "Reconstruction stopped successfully."}
        except Exception as e:
            logger.error(f"Error stopping reconstruction: {str(e)}")
            return {"status": "Error stopping reconstruction", "error": str(e)}

    return stop_reconstruction
