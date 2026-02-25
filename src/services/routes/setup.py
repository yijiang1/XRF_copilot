from fastapi import APIRouter, HTTPException
from ..models import XRFSimulationParams

router = APIRouter()


def setup_endpoint(simulation_output):
    """Register the setup simulation endpoint."""

    @router.post("/setup_simulation/")
    async def setup_simulation(input_dict: XRFSimulationParams):
        try:
            params = input_dict.model_dump()
            simulation_output["params"] = params
            simulation_output["error"] = None
            return {
                "status": "Simulation setup completed successfully.",
                "params": params,
            }
        except Exception as e:
            simulation_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return setup_simulation
