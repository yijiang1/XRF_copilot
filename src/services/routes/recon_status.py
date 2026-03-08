from fastapi import APIRouter

router = APIRouter()


def _latest_session(session_manager, method: str):
    with session_manager._lock:
        candidates = [s for s in session_manager._sessions.values() if s.method == method]
    return max(candidates, key=lambda s: s.created_at) if candidates else None


def recon_status_endpoints(session_manager):
    """Register all Panpan reconstruction status endpoints (session-aware)."""

    @router.post("/get_recon_status/")
    async def get_recon_status(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Panpan")
        if s is None:
            return {"status": "Not running", "error": None, "is_running": False}
        return {
            "status": "Running" if s.process_status["is_running"] else "Not running",
            "error": s.output.get("error"),
            "is_running": s.process_status["is_running"],
        }

    @router.post("/get_recon_worker_status/")
    async def get_recon_worker_status(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Panpan")
        if s is None:
            return {"is_running": False, "worker_logs": []}
        ps = s.process_status
        return {
            "is_running": ps["is_running"],
            "worker_logs": s.worker_logs,
            "process_pid": ps["process"].pid if ps["process"] else None,
            "process_alive": ps["process"].is_alive() if ps["process"] else False,
        }

    @router.post("/get_recon_progress/")
    async def get_recon_progress(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Panpan")
        if s is None:
            return {"current_epoch": 0, "total_epochs": 0, "progress_percent": 0, "is_running": False}
        out = s.output
        cur, tot = out.get("current_epoch", 0), out.get("total_epochs", 0)
        return {
            "current_epoch": cur,
            "total_epochs": tot,
            "progress_percent": (cur / tot * 100.0) if tot > 0 else 0.0,
            "is_running": s.process_status["is_running"],
        }

    @router.post("/get_recon_results/")
    async def get_recon_results(session_id: str = ""):
        s = session_manager.get_session(session_id) if session_id else _latest_session(session_manager, "Panpan")
        if s is None:
            return {"results_ready": False, "recon_file": "", "is_running": False}
        recon_file = s.output.get("recon_file", "")
        return {"results_ready": bool(recon_file), "recon_file": recon_file, "is_running": s.process_status["is_running"]}

    return {"get_recon_status": get_recon_status, "get_recon_worker_status": get_recon_worker_status,
            "get_recon_progress": get_recon_progress, "get_recon_results": get_recon_results}
