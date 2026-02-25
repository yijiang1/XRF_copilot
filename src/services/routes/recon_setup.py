from fastapi import APIRouter, HTTPException
from ..models import XRFReconstructionParams

router = APIRouter()


def recon_setup_endpoint(recon_output):
    """Register the setup reconstruction endpoint."""

    @router.post("/setup_reconstruction/")
    async def setup_reconstruction(input_dict: XRFReconstructionParams):
        try:
            params = input_dict.model_dump()
            recon_output["params"] = params
            recon_output["error"] = None
            return {
                "status": "Reconstruction setup completed successfully.",
                "params": params,
            }
        except Exception as e:
            recon_output["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    return setup_reconstruction
