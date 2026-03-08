from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..fl_worker import fl_correction_worker_process
from ..queue_handlers import monitor_session_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def fl_run_endpoint(session_manager, fl_setup_output):
    """Register the run FL correction endpoint (session-based)."""

    @router.post("/run_fl_correction/")
    async def run_fl_correction():
        try:
            params = fl_setup_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_fl_correction/ first.",
                )

            # Create a new session
            session = session_manager.create_session("BNL", params)
            session.output["params"] = params

            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=fl_correction_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(
                f"FL correction worker started — session {session.session_id}, PID {process.pid}"
            )

            session.process_status["is_running"] = True
            session.process_status["process"] = process
            session.process_status["status_queue"] = status_queue
            session.process_status["stop_event"] = stop_event

            monitor_thread = threading.Thread(
                target=monitor_session_queue,
                args=(session,),
                daemon=True,
            )
            monitor_thread.start()

            return {
                "status": "FL correction started in background.",
                "session_id": session.session_id,
                "display_name": session.display_name,
            }

        except Exception as e:
            logger.error(f"Error starting FL correction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return run_fl_correction
