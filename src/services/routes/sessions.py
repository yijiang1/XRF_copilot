"""Session management endpoints (list, remove, clear)."""

from fastapi import APIRouter

router = APIRouter()


def session_endpoints(session_manager):
    """Register session management endpoints against the shared session_manager."""

    @router.get("/list_sessions/")
    async def list_sessions():
        """Return lightweight summaries of all sessions, sorted oldest-first."""
        return session_manager.list_sessions()

    @router.post("/remove_session/")
    async def remove_session(session_id: str):
        """Remove a completed (non-running) session."""
        success = session_manager.remove_session(session_id)
        return {
            "status": "removed" if success else "not_found_or_running",
            "session_id": session_id,
        }

    @router.post("/clear_finished_sessions/")
    async def clear_finished_sessions():
        """Remove all completed sessions."""
        count = session_manager.clear_finished()
        return {"status": "cleared", "count": count}

    @router.get("/get_session_status/")
    async def get_session_status(session_id: str):
        """Return the full summary for a single session."""
        session = session_manager.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}
        return session.summary()

    @router.get("/get_session_worker_logs/")
    async def get_session_worker_logs(session_id: str):
        """Return the worker logs for a single session."""
        session = session_manager.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found", "logs": []}
        return {"session_id": session_id, "logs": session.worker_logs}

    @router.post("/stop_session/")
    async def stop_session(session_id: str):
        """Stop a specific running session."""
        import time as _time
        session = session_manager.get_session(session_id)
        if session is None:
            return {"status": f"Session {session_id!r} not found"}
        ps = session.process_status
        if not ps["is_running"]:
            return {"status": "Session is not running"}
        try:
            if ps["stop_event"]:
                ps["stop_event"].set()
            _time.sleep(0.5)
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
            return {"status": f"Session {session_id} stopped successfully"}
        except Exception as e:
            return {"status": "Error stopping session", "error": str(e)}

    return list_sessions, remove_session, clear_finished_sessions, get_session_status
