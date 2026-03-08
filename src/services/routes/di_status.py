from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def _latest_session(session_manager, method: str):
    with session_manager._lock:
        candidates = [s for s in session_manager._sessions.values() if s.method == method]
    return max(candidates, key=lambda s: s.created_at) if candidates else None


def di_status_endpoints(session_manager):
    """Register Di reconstruction status endpoints (session-aware)."""

    @router.post("/get_di_recon_status/")
    async def get_di_recon_status(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Wendy")
        if s is None:
            return {"error": None, "output": "", "is_running": False}
        return {"error": s.output.get("error"), "output": s.output.get("recon_file", ""), "is_running": s.process_status["is_running"]}

    @router.post("/get_di_recon_progress/")
    async def get_di_recon_progress(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Wendy")
        if s is None:
            return {"current_epoch": 0, "total_epochs": 0, "is_running": False}
        out = s.output
        return {
            "current_epoch": out.get("current_epoch", 0),
            "total_epochs": out.get("total_epochs", 0),
            "is_running": s.process_status["is_running"],
        }

    @router.post("/get_di_recon_results/")
    async def get_di_recon_results(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Wendy")
        if s is None:
            return {"recon_file": "", "error": None}
        return {"recon_file": s.output.get("recon_file", ""), "error": s.output.get("error")}

    @router.post("/get_di_recon_worker_status/")
    async def get_di_recon_worker_status(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Wendy")
        if s is None:
            return {"logs": [], "latest_status": None}
        return {"logs": list(s.worker_logs[-200:]), "latest_status": s.latest_worker_status.get("status")}

    return get_di_recon_status, get_di_recon_progress, get_di_recon_results, get_di_recon_worker_status
