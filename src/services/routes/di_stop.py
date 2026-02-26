from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def di_stop_endpoint(di_process_status, di_output, di_latest_worker_status):
    """Register the stop Di reconstruction endpoint."""

    @router.post("/stop_di_reconstruction/")
    async def stop_di_reconstruction():
        if not di_process_status["is_running"]:
            return {"status": "No Di reconstruction is currently running."}

        try:
            if di_process_status["stop_event"]:
                di_process_status["stop_event"].set()

            if di_process_status["process"] and di_process_status["process"].is_alive():
                di_process_status["process"].terminate()
                di_process_status["process"].join(timeout=3)
                if di_process_status["process"].is_alive():
                    di_process_status["process"].kill()

            di_process_status["is_running"] = False
            di_process_status["process"] = None
            di_process_status["status_queue"] = None
            di_process_status["stop_event"] = None
            di_latest_worker_status["status"] = "stopped"

            return {"status": "Di reconstruction stopped successfully."}

        except Exception as e:
            logger.error(f"Error stopping Di reconstruction: {str(e)}")
            return {"status": f"Error stopping: {str(e)}"}

    return stop_di_reconstruction
