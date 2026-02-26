"""FastAPI backend for XRF Simulation."""

import os
import sys
import logging
import multiprocessing
import atexit

# Set start method for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .routes import setup, run, stop, status
from .routes import recon_setup, recon_run, recon_stop, recon_status
from .routes import fl_setup, fl_run, fl_stop, fl_status
from .routes import di_setup, di_run, di_stop, di_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(title="XRF Simulation API", description="API for XRF Fluorescence Simulation")

# API key protection
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Reject requests without a valid API key (if one is configured)."""
    if not BACKEND_API_KEY:
        return await call_next(request)
    if request.url.path == "/":
        return await call_next(request)
    import secrets
    provided_key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided_key, BACKEND_API_KEY):
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid or missing API key"},
        )
    return await call_next(request)


# Shared state for process communication
process_status = {
    "is_running": False,
    "process": None,
    "status_queue": None,
    "stop_event": None,
}

simulation_output = {
    "params": None,
    "error": None,
    "current_batch": 0,
    "total_batches": 0,
    "sim_xrf_file": "",
    "sim_xrt_file": "",
}

worker_logs = []
latest_worker_status = {"timestamp": 0, "status": None}

# Shared state for reconstruction process communication
recon_process_status = {
    "is_running": False,
    "process": None,
    "status_queue": None,
    "stop_event": None,
}

recon_output = {
    "params": None,
    "error": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "recon_file": "",
}

recon_worker_logs = []
recon_latest_worker_status = {"timestamp": 0, "status": None}

# Shared state for FL correction process
fl_process_status = {
    "is_running": False,
    "process": None,
    "status_queue": None,
    "stop_event": None,
}

fl_output = {
    "params": None,
    "error": None,
    "current_step": 0,
    "total_steps": 0,
    "step_label": "",
    "recon_file": "",
}

fl_worker_logs = []
fl_latest_worker_status = {"timestamp": 0, "status": None}

# Shared state for Di et al. reconstruction process
di_process_status = {
    "is_running": False,
    "process": None,
    "status_queue": None,
    "stop_event": None,
}

di_output = {
    "params": None,
    "error": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "recon_file": "",
}

di_worker_logs = []
di_latest_worker_status = {"timestamp": 0, "status": None}


def cleanup_resources():
    """Clean up worker process on shutdown."""
    logger.info("Cleaning up resources on shutdown")
    if (
        process_status["is_running"]
        and process_status["process"]
        and process_status["process"].is_alive()
    ):
        if process_status["stop_event"]:
            process_status["stop_event"].set()
        process_status["process"].terminate()
        process_status["process"].join(timeout=2)
        if process_status["process"].is_alive():
            process_status["process"].kill()
        process_status["is_running"] = False
        process_status["process"] = None
        process_status["status_queue"] = None
        process_status["stop_event"] = None

    if (
        recon_process_status["is_running"]
        and recon_process_status["process"]
        and recon_process_status["process"].is_alive()
    ):
        if recon_process_status["stop_event"]:
            recon_process_status["stop_event"].set()
        recon_process_status["process"].terminate()
        recon_process_status["process"].join(timeout=2)
        if recon_process_status["process"].is_alive():
            recon_process_status["process"].kill()
        recon_process_status["is_running"] = False
        recon_process_status["process"] = None
        recon_process_status["status_queue"] = None
        recon_process_status["stop_event"] = None

    if (
        fl_process_status["is_running"]
        and fl_process_status["process"]
        and fl_process_status["process"].is_alive()
    ):
        if fl_process_status["stop_event"]:
            fl_process_status["stop_event"].set()
        fl_process_status["process"].terminate()
        fl_process_status["process"].join(timeout=2)
        if fl_process_status["process"].is_alive():
            fl_process_status["process"].kill()
        fl_process_status["is_running"] = False
        fl_process_status["process"] = None
        fl_process_status["status_queue"] = None
        fl_process_status["stop_event"] = None

    if (
        di_process_status["is_running"]
        and di_process_status["process"]
        and di_process_status["process"].is_alive()
    ):
        if di_process_status["stop_event"]:
            di_process_status["stop_event"].set()
        di_process_status["process"].terminate()
        di_process_status["process"].join(timeout=2)
        if di_process_status["process"].is_alive():
            di_process_status["process"].kill()
        di_process_status["is_running"] = False
        di_process_status["process"] = None
        di_process_status["status_queue"] = None
        di_process_status["stop_event"] = None

    logger.info("Cleanup complete")


@app.on_event("shutdown")
async def on_shutdown():
    cleanup_resources()


atexit.register(cleanup_resources)

# Initialize simulation routes
setup.setup_endpoint(simulation_output)
run.run_endpoint(process_status, simulation_output, latest_worker_status, worker_logs)
stop.stop_endpoint(process_status, simulation_output, latest_worker_status)
status.status_endpoints(process_status, simulation_output, latest_worker_status, worker_logs)

# Initialize reconstruction routes
recon_setup.recon_setup_endpoint(recon_output)
recon_run.recon_run_endpoint(recon_process_status, recon_output, recon_latest_worker_status, recon_worker_logs)
recon_stop.recon_stop_endpoint(recon_process_status, recon_output, recon_latest_worker_status)
recon_status.recon_status_endpoints(recon_process_status, recon_output, recon_latest_worker_status, recon_worker_logs)

# Initialize FL correction routes
fl_setup.fl_setup_endpoint(fl_output)
fl_run.fl_run_endpoint(fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs)
fl_stop.fl_stop_endpoint(fl_process_status, fl_output, fl_latest_worker_status)
fl_status.fl_status_endpoints(fl_process_status, fl_output, fl_latest_worker_status, fl_worker_logs)

# Initialize Di et al. reconstruction routes
di_setup.di_setup_endpoint(di_output)
di_run.di_run_endpoint(di_process_status, di_output, di_latest_worker_status, di_worker_logs)
di_stop.di_stop_endpoint(di_process_status, di_output, di_latest_worker_status)
di_status.di_status_endpoints(di_process_status, di_output, di_latest_worker_status, di_worker_logs)

# Include routers
app.include_router(setup.router, tags=["Simulation"])
app.include_router(run.router, tags=["Simulation"])
app.include_router(stop.router, tags=["Simulation"])
app.include_router(status.router, tags=["Status"])
app.include_router(recon_setup.router, tags=["Reconstruction"])
app.include_router(recon_run.router, tags=["Reconstruction"])
app.include_router(recon_stop.router, tags=["Reconstruction"])
app.include_router(recon_status.router, tags=["Reconstruction"])
app.include_router(fl_setup.router, tags=["FL Correction (BNL)"])
app.include_router(fl_run.router, tags=["FL Correction (BNL)"])
app.include_router(fl_stop.router, tags=["FL Correction (BNL)"])
app.include_router(fl_status.router, tags=["FL Correction (BNL)"])
app.include_router(di_setup.router, tags=["Reconstruction (Di et al.)"])
app.include_router(di_run.router, tags=["Reconstruction (Di et al.)"])
app.include_router(di_stop.router, tags=["Reconstruction (Di et al.)"])
app.include_router(di_status.router, tags=["Reconstruction (Di et al.)"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to the XRF Copilot API",
        "simulation_endpoints": {
            "setup_simulation": "POST /setup_simulation/",
            "run_simulation": "POST /run_simulation/",
            "stop_simulation": "POST /stop_simulation/",
            "get_simulation_status": "POST /get_simulation_status/",
            "get_worker_status": "POST /get_worker_status/",
            "get_progress": "POST /get_progress/",
            "get_results": "POST /get_results/",
        },
        "reconstruction_endpoints": {
            "setup_reconstruction": "POST /setup_reconstruction/",
            "run_reconstruction": "POST /run_reconstruction/",
            "stop_reconstruction": "POST /stop_reconstruction/",
            "get_recon_status": "POST /get_recon_status/",
            "get_recon_worker_status": "POST /get_recon_worker_status/",
            "get_recon_progress": "POST /get_recon_progress/",
            "get_recon_results": "POST /get_recon_results/",
        },
    }


def cli():
    """CLI entry point for the backend server."""
    import argparse
    import uvicorn

    global BACKEND_API_KEY

    parser = argparse.ArgumentParser(description="XRF Simulation Backend API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--api-key", type=str, default=None, help="Set an API key")
    args = parser.parse_args()

    if args.api_key:
        BACKEND_API_KEY = args.api_key
        logger.info("API key set via --api-key flag.")

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    cli()
