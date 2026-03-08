from fastapi import APIRouter
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()


def _latest_session(session_manager, method: str):
    with session_manager._lock:
        candidates = [s for s in session_manager._sessions.values() if s.method == method]
    return max(candidates, key=lambda s: s.created_at) if candidates else None


def fl_stop_endpoint(session_manager):
    """Register the stop FL correction endpoint (session-aware)."""

    @router.post("/stop_fl_correction/")
    async def stop_fl_correction(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "BNL")
        if s is None or not s.process_status["is_running"]:
            return {"status": "No FL correction is running."}
        try:
            ps = s.process_status
            logger.info(f"Stopping FL session {s.session_id}")
            if ps["stop_event"]:
                ps["stop_event"].set()
            time.sleep(0.5)
            if ps["process"] and ps["process"].is_alive():
                ps["process"].kill()
                ps["process"].join(timeout=5)
            if ps["status_queue"]:
                try:
                    import queue
                    while True:
                        ps["status_queue"].get(block=False)
                except queue.Empty:
                    pass
            ps["is_running"] = False
            ps["process"] = None
            ps["status_queue"] = None
            ps["stop_event"] = None
            return {"status": "FL correction stopped successfully."}
        except Exception as e:
            logger.error(f"Error stopping FL correction: {e}")
            return {"status": "Error stopping FL correction", "error": str(e)}

    return stop_fl_correction
