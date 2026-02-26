from fastapi import APIRouter, HTTPException
from ..models import DiReconParams

router = APIRouter()


def di_setup_endpoint(di_output):
    """Register the setup Di reconstruction endpoint."""

    @router.post("/setup_di_reconstruction/")
    async def setup_di_reconstruction(input_dict: DiReconParams):
        try:
            params = input_dict.model_dump()
            di_output["params"] = params
            di_output["error"] = None
            return {
                "status": "Di reconstruction setup completed successfully.",
                "params": params,
            }
        except Exception as e:
            di_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return setup_di_reconstruction
