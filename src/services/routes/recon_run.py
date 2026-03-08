from fastapi import APIRouter, HTTPException
import threading
import logging
from multiprocessing import Process, Queue, Event
from ..recon_worker import reconstruction_worker_process
from ..queue_handlers import monitor_session_queue

logger = logging.getLogger(__name__)

router = APIRouter()


def recon_run_endpoint(session_manager, recon_setup_output):
    """Register the run Panpan reconstruction endpoint (session-based)."""

    @router.post("/run_reconstruction/")
    async def run_reconstruction():
        try:
            params = recon_setup_output.get("params")
            if not params:
                raise HTTPException(
                    status_code=400,
                    detail="No parameters set. Call /setup_reconstruction/ first.",
                )

            # Create a new session
            session = session_manager.create_session("Panpan", params)
            session.output["params"] = params

            status_queue = Queue()
            stop_event = Event()

            process = Process(
                target=reconstruction_worker_process,
                args=(params, status_queue, stop_event),
            )
            process.daemon = True
            process.start()
            logger.info(
                f"Panpan reconstruction worker started — session {session.session_id}, PID {process.pid}"
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
                "status": "Reconstruction started in background.",
                "session_id": session.session_id,
                "display_name": session.display_name,
            }

        except Exception as e:
            logger.error(f"Error starting reconstruction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return run_reconstruction
