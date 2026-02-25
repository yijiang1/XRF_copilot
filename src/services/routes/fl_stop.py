from fastapi import APIRouter
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()


def fl_stop_endpoint(fl_process_status, fl_output, fl_latest_worker_status):
    """Register the stop FL correction endpoint."""

    @router.post("/stop_fl_correction/")
    async def stop_fl_correction():
        if not fl_process_status["is_running"]:
            return {"status": "No FL correction is running."}

        try:
            logger.info("Stopping FL correction process")

            if fl_process_status["stop_event"]:
                fl_process_status["stop_event"].set()

            time.sleep(0.5)

            if fl_process_status["process"] and fl_process_status["process"].is_alive():
                fl_process_status["process"].kill()
                fl_process_status["process"].join(timeout=5)

            # Drain queue
            if fl_process_status["status_queue"]:
                try:
                    while True:
                        fl_process_status["status_queue"].get(block=False)
                except Exception:
                    pass

            # Reset state
            fl_process_status["is_running"] = False
            fl_process_status["process"] = None
            fl_process_status["status_queue"] = None
            fl_process_status["stop_event"] = None

            fl_output["error"] = None
            fl_output["current_step"] = 0
            fl_output["total_steps"] = 0
            fl_output["step_label"] = ""
            fl_latest_worker_status["timestamp"] = 0
            fl_latest_worker_status["status"] = None

            return {"status": "FL correction stopped successfully."}
        except Exception as e:
            logger.error(f"Error stopping FL correction: {str(e)}")
            return {"status": "Error stopping FL correction", "error": str(e)}

    return stop_fl_correction
