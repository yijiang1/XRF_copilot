"""Session manager for multiple concurrent XRF reconstruction sessions."""

import threading
import time
import uuid
import logging

logger = logging.getLogger(__name__)

MAX_COMPLETED_SESSIONS = 20   # auto-evict oldest finished sessions
MAX_WORKER_LOGS = 1000        # cap per-session log buffer


class XRFSession:
    """State for a single reconstruction session."""

    __slots__ = (
        "session_id",
        "display_name",
        "method",        # "BNL" | "Panpan" | "Wendy"
        "params",
        "process_status",
        "output",
        "worker_logs",
        "latest_worker_status",
        "created_at",
    )

    def __init__(self, session_id: str, method: str, params: dict):
        self.session_id = session_id
        self.method = method
        self.params = dict(params)

        gpu = params.get("gpu_id", "?")
        self.display_name = f"{method} GPU:{gpu}"

        self.process_status = {
            "is_running": False,
            "process": None,
            "status_queue": None,
            "stop_event": None,
        }

        if method == "BNL":
            self.output = {
                "params": None,
                "error": None,
                "current_step": 0,
                "total_steps": 0,
                "step_label": "",
                "recon_file": "",
            }
        else:  # Panpan or Wendy
            self.output = {
                "params": None,
                "error": None,
                "current_epoch": 0,
                "total_epochs": 0,
                "recon_file": "",
            }

        self.worker_logs: list = []
        self.latest_worker_status: dict = {"timestamp": 0, "status": None}
        self.created_at: float = time.time()

    def summary(self) -> dict:
        """Lightweight summary for the list_sessions endpoint."""
        out = self.output
        if self.method == "BNL":
            current = out.get("current_step", 0)
            total = out.get("total_steps", 0)
        else:
            current = out.get("current_epoch", 0)
            total = out.get("total_epochs", 0)

        progress_percent = (current / total * 100.0) if total > 0 else 0.0

        return {
            "session_id": self.session_id,
            "display_name": self.display_name,
            "method": self.method,
            "is_running": self.process_status["is_running"],
            "progress_percent": progress_percent,
            "current": current,
            "total": total,
            "error": out.get("error"),
            "recon_file": out.get("recon_file", ""),
            "created_at": self.created_at,
        }


class XRFSessionManager:
    """Thread-safe manager for multiple concurrent XRF reconstruction sessions."""

    def __init__(self):
        self._sessions: dict[str, XRFSession] = {}
        self._lock = threading.Lock()

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self, method: str, params: dict) -> XRFSession:
        """Create a new session and register it. Returns the new session."""
        session_id = uuid.uuid4().hex[:12]
        session = XRFSession(session_id, method, params)
        with self._lock:
            self._sessions[session_id] = session
            self._evict_old_sessions()
        logger.info(f"Created session {session_id}: {session.display_name}")
        return session

    def _evict_old_sessions(self):
        """Remove oldest completed sessions when the count exceeds the limit.

        Must be called while holding ``_lock``.
        """
        completed = [
            s for s in self._sessions.values()
            if not s.process_status["is_running"]
        ]
        if len(completed) <= MAX_COMPLETED_SESSIONS:
            return
        completed.sort(key=lambda s: s.created_at)
        for s in completed[:len(completed) - MAX_COMPLETED_SESSIONS]:
            del self._sessions[s.session_id]
            logger.info(f"Auto-evicted old session {s.session_id} ({s.display_name})")

    def get_session(self, session_id: str) -> "XRFSession | None":
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """Return lightweight summaries of all sessions, sorted by creation time."""
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.created_at)
            return [s.summary() for s in sessions]

    def remove_session(self, session_id: str) -> bool:
        """Remove a non-running session. Returns False if running or not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.process_status["is_running"]:
                return False
            del self._sessions[session_id]
            logger.info(f"Removed session {session_id}")
            return True

    def clear_finished(self) -> int:
        """Remove all non-running sessions. Returns count removed."""
        with self._lock:
            finished = [
                sid for sid, s in self._sessions.items()
                if not s.process_status["is_running"]
            ]
            for sid in finished:
                del self._sessions[sid]
        logger.info(f"Cleared {len(finished)} finished session(s)")
        return len(finished)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def stop_all(self):
        """Terminate all running sessions. Called during server shutdown."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            ps = session.process_status
            if ps["is_running"] and ps["process"] and ps["process"].is_alive():
                logger.info(f"Stopping session {session.session_id} on shutdown")
                if ps["stop_event"]:
                    ps["stop_event"].set()
                ps["process"].terminate()
                ps["process"].join(timeout=2)
                if ps["process"].is_alive():
                    ps["process"].kill()
                ps["is_running"] = False
