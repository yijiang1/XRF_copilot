from fastapi import APIRouter

router = APIRouter()


def recon_status_endpoints(recon_process_status, recon_output, recon_latest_worker_status, recon_worker_logs):
    """Register all reconstruction status endpoints."""

    @router.post("/get_recon_status/")
    async def get_recon_status():
        return {
            "status": "Running" if recon_process_status["is_running"] else "Not running",
            "error": recon_output.get("error"),
            "is_running": recon_process_status["is_running"],
        }

    @router.post("/get_recon_worker_status/")
    async def get_recon_worker_status():
        return {
            "is_running": recon_process_status["is_running"],
            "worker_logs": recon_worker_logs,
            "process_pid": (
                recon_process_status["process"].pid if recon_process_status["process"] else None
            ),
            "process_alive": (
                recon_process_status["process"].is_alive()
                if recon_process_status["process"]
                else False
            ),
        }

    @router.post("/get_recon_progress/")
    async def get_recon_progress():
        current_epoch = recon_output.get("current_epoch", 0)
        total_epochs = recon_output.get("total_epochs", 0)

        progress_percent = 0
        if total_epochs > 0:
            progress_percent = (current_epoch / total_epochs) * 100

        return {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "progress_percent": progress_percent,
            "is_running": recon_process_status["is_running"],
        }

    @router.post("/get_recon_results/")
    async def get_recon_results():
        recon_file = recon_output.get("recon_file", "")
        results_ready = bool(recon_file)

        return {
            "results_ready": results_ready,
            "recon_file": recon_file,
            "is_running": recon_process_status["is_running"],
        }

    return {
        "get_recon_status": get_recon_status,
        "get_recon_worker_status": get_recon_worker_status,
        "get_recon_progress": get_recon_progress,
        "get_recon_results": get_recon_results,
    }
