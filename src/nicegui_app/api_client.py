"""Async HTTP client wrapping all backend API endpoints."""

import httpx
from .config import API_ENDPOINT


class XRFSimulationAPIClient:
    """Async HTTP client for the XRF Simulation backend."""

    def __init__(self, endpoint: str = API_ENDPOINT, api_key: str = ""):
        self.endpoint = endpoint
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def set_api_key(self, api_key: str):
        self._api_key = api_key

    def _headers(self) -> dict:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        return {}

    async def setup_simulation(self, params: dict) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/setup_simulation/",
            json=params,
            headers=self._headers(),
        )
        if resp.status_code == 422:
            detail = resp.json()
            raise Exception(f"Validation error: {detail}")
        resp.raise_for_status()
        return resp.json()

    async def run_simulation(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/run_simulation/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_simulation(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/stop_simulation/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_progress(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_progress/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_simulation_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_simulation_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_results(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_results/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_worker_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_worker_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # ── Reconstruction endpoints ──────────────────────────────────────────────

    async def setup_reconstruction(self, params: dict) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/setup_reconstruction/",
            json=params,
            headers=self._headers(),
        )
        if resp.status_code == 422:
            detail = resp.json()
            raise Exception(f"Validation error: {detail}")
        resp.raise_for_status()
        return resp.json()

    async def run_reconstruction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/run_reconstruction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_reconstruction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/stop_reconstruction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recon_progress(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_recon_progress/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recon_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_recon_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recon_results(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_recon_results/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recon_worker_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_recon_worker_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # ── FL Correction (BNL) endpoints ─────────────────────────────────────────

    async def setup_fl_correction(self, params: dict) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/setup_fl_correction/",
            json=params,
            headers=self._headers(),
        )
        if resp.status_code == 422:
            raise Exception(f"Validation error: {resp.json()}")
        resp.raise_for_status()
        return resp.json()

    async def run_fl_correction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/run_fl_correction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_fl_correction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/stop_fl_correction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_fl_progress(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_fl_progress/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_fl_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_fl_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_fl_results(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_fl_results/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_fl_worker_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_fl_worker_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # ── Di et al. 2017 reconstruction endpoints ───────────────────────────────

    async def setup_di_reconstruction(self, params: dict) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/setup_di_reconstruction/",
            json=params,
            headers=self._headers(),
        )
        if resp.status_code == 422:
            raise Exception(f"Validation error: {resp.json()}")
        resp.raise_for_status()
        return resp.json()

    async def run_di_reconstruction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/run_di_reconstruction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_di_reconstruction(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/stop_di_reconstruction/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_di_recon_progress(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_di_recon_progress/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_di_recon_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_di_recon_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_di_recon_results(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_di_recon_results/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def get_di_recon_worker_status(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/get_di_recon_worker_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # ── Reconstruction image endpoints ─────────────────────────────────────────

    async def get_session_recon_info(self, session_id: str) -> dict:
        resp = await self._client.get(
            f"{self.endpoint}/get_session_recon_info/",
            params={"session_id": session_id},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recon_slice(
        self, session_id: str, elem_idx: int = 0, slice_idx: int = 0,
        file: str = "",
    ) -> dict:
        params = {
            "session_id": session_id,
            "elem_idx": elem_idx,
            "slice_idx": slice_idx,
        }
        if file:
            params["file"] = file
        resp = await self._client.get(
            f"{self.endpoint}/get_recon_slice/",
            params=params,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    # ── GPU & Session endpoints ───────────────────────────────────────────────

    async def gpu_status(self) -> dict:
        resp = await self._client.get(
            f"{self.endpoint}/gpu_status/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def list_sessions(self) -> list:
        resp = await self._client.get(
            f"{self.endpoint}/list_sessions/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def remove_session(self, session_id: str) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/remove_session/",
            params={"session_id": session_id},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_session(self, session_id: str) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/stop_session/",
            params={"session_id": session_id},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def clear_finished_sessions(self) -> dict:
        resp = await self._client.post(
            f"{self.endpoint}/clear_finished_sessions/", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()
