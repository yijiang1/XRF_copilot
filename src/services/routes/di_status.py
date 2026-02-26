from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def di_status_endpoints(di_process_status, di_output, di_latest_worker_status, di_worker_logs):
    """Register Di reconstruction status endpoints."""

    @router.post("/get_di_recon_status/")
    async def get_di_recon_status():
        return {
            "error": di_output.get("error"),
            "output": di_output.get("recon_file", ""),
        }

    @router.post("/get_di_recon_progress/")
    async def get_di_recon_progress():
        return {
            "current_epoch": di_output.get("current_epoch", 0),
            "total_epochs": di_output.get("total_epochs", 0),
            "is_running": di_process_status["is_running"],
        }

    @router.post("/get_di_recon_results/")
    async def get_di_recon_results():
        return {
            "recon_file": di_output.get("recon_file", ""),
            "error": di_output.get("error"),
        }

    @router.post("/get_di_recon_worker_status/")
    async def get_di_recon_worker_status():
        return {
            "logs": list(di_worker_logs[-200:]),
            "latest_status": di_latest_worker_status.get("status"),
        }

    return (
        get_di_recon_status,
        get_di_recon_progress,
        get_di_recon_results,
        get_di_recon_worker_status,
    )
