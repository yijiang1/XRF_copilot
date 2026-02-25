from fastapi import APIRouter

router = APIRouter()


def fl_status_endpoints(fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs):
    """Register all FL correction status endpoints."""

    @router.post("/get_fl_status/")
    async def get_fl_status():
        return {
            "status": "Running" if fl_process_status["is_running"] else "Not running",
            "error": fl_output.get("error"),
            "is_running": fl_process_status["is_running"],
        }

    @router.post("/get_fl_worker_status/")
    async def get_fl_worker_status():
        return {
            "is_running": fl_process_status["is_running"],
            "worker_logs": fl_worker_logs,
            "process_pid": (
                fl_process_status["process"].pid if fl_process_status["process"] else None
            ),
            "process_alive": (
                fl_process_status["process"].is_alive()
                if fl_process_status["process"]
                else False
            ),
        }

    @router.post("/get_fl_progress/")
    async def get_fl_progress():
        current_step = fl_output.get("current_step", 0)
        total_steps = fl_output.get("total_steps", 0)
        progress_percent = 0.0
        if total_steps > 0:
            progress_percent = (current_step / total_steps) * 100.0
        return {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
            "step_label": fl_output.get("step_label", ""),
            "is_running": fl_process_status["is_running"],
        }

    @router.post("/get_fl_results/")
    async def get_fl_results():
        recon_file = fl_output.get("recon_file", "")
        results_ready = bool(recon_file)
        return {
            "results_ready": results_ready,
            "recon_file": recon_file,
            "is_running": fl_process_status["is_running"],
        }

    return {
        "get_fl_status": get_fl_status,
        "get_fl_worker_status": get_fl_worker_status,
        "get_fl_progress": get_fl_progress,
        "get_fl_results": get_fl_results,
    }
