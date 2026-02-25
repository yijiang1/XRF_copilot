from fastapi import APIRouter
import time

router = APIRouter()


def status_endpoints(process_status, simulation_output, latest_worker_status, worker_logs):
    """Register all status-related endpoints."""

    @router.post("/get_simulation_status/")
    async def get_simulation_status():
        return {
            "status": "Running" if process_status["is_running"] else "Not running",
            "error": simulation_output.get("error"),
            "is_running": process_status["is_running"],
        }

    @router.post("/get_worker_status/")
    async def get_worker_status():
        return {
            "is_running": process_status["is_running"],
            "worker_logs": worker_logs,
            "process_pid": (
                process_status["process"].pid if process_status["process"] else None
            ),
            "process_alive": (
                process_status["process"].is_alive()
                if process_status["process"]
                else False
            ),
        }

    @router.post("/get_progress/")
    async def get_progress():
        current_batch = simulation_output.get("current_batch", 0)
        total_batches = simulation_output.get("total_batches", 0)

        progress_percent = 0
        if total_batches > 0:
            progress_percent = (current_batch / total_batches) * 100

        return {
            "current_batch": current_batch,
            "total_batches": total_batches,
            "progress_percent": progress_percent,
            "is_running": process_status["is_running"],
        }

    @router.post("/get_results/")
    async def get_results():
        sim_xrf_file = simulation_output.get("sim_xrf_file", "")
        sim_xrt_file = simulation_output.get("sim_xrt_file", "")
        results_ready = bool(sim_xrf_file and sim_xrt_file)

        return {
            "results_ready": results_ready,
            "sim_xrf_file": sim_xrf_file,
            "sim_xrt_file": sim_xrt_file,
            "is_running": process_status["is_running"],
        }

    return {
        "get_simulation_status": get_simulation_status,
        "get_worker_status": get_worker_status,
        "get_progress": get_progress,
        "get_results": get_results,
    }
