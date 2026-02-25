from fastapi import APIRouter, HTTPException
from ..models import FLCorrectionParams

router = APIRouter()


def fl_setup_endpoint(fl_output):
    """Register the setup FL correction endpoint."""

    @router.post("/setup_fl_correction/")
    async def setup_fl_correction(input_dict: FLCorrectionParams):
        try:
            params = input_dict.model_dump()
            fl_output["params"] = params
            fl_output["error"] = None
            return {
                "status": "FL correction setup completed successfully.",
                "params": params,
            }
        except Exception as e:
            fl_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return setup_fl_correction
