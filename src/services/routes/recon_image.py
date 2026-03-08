"""Serve reconstruction checkpoint slices as base64 PNG data URLs.

Endpoints:
    GET /get_session_recon_info/  — metadata from latest checkpoint
    GET /get_recon_slice/         — one 2D slice as Viridis PNG
"""

import os
import io
import glob
import base64

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter

router = APIRouter()


def _read_recon_slice(filepath: str, elem_idx: int, slice_idx: int) -> dict:
    """Read one 2D slice from a reconstruction HDF5 and render as base64 PNG."""
    with h5py.File(filepath, "r", locking=False) as f:
        arr = np.array(f["densities"][elem_idx, slice_idx, :, :])

    vmin, vmax, vmean = float(arr.min()), float(arr.max()), float(arr.mean())
    ny, nx = arr.shape

    p2 = float(np.nanpercentile(arr, 2))
    p98 = float(np.nanpercentile(arr, 98))
    if p98 <= p2:
        p98 = p2 + 1.0

    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap="viridis", vmin=p2, vmax=p98, format="png")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "data_url": data_url,
        "vmin": vmin,
        "vmax": vmax,
        "vmean": vmean,
        "ny": ny,
        "nx": nx,
    }


def _find_latest_checkpoint(session):
    """Find the latest reconstruction HDF5 for a session.

    Prefers the most recently modified file (checkpoint or in-place recon.h5).
    This allows the viewer to show live updates between checkpoint saves.

    All methods: try queue-based fast path first (session.output["latest_checkpoint"]),
    then fall back to directory scan filtered by session creation time.
    """
    method = getattr(session, "method", "")

    # Queue-based fast path: worker announces checkpoints via status queue
    ckpt = session.output.get("latest_checkpoint")
    if ckpt:
        f = ckpt.get("file", "")
        if f and os.path.isfile(f):
            return ckpt

    # Scan output directory for reconstruction HDF5 files
    recon_path = session.output.get("recon_path", "")
    if not recon_path:
        return None

    if method == "BNL":
        ckpt_dir = os.path.join(recon_path, "recon")
    else:
        ckpt_dir = recon_path

    pattern = os.path.join(ckpt_dir, "recon_*.h5")
    session_start = getattr(session, "created_at", 0)
    all_files = glob.glob(pattern)

    # Keep only numbered checkpoints: recon_{N}.h5
    # Exclude recon.h5, recon_initial.h5, recon_{N}_ending_condition.h5
    _EXCLUDE = {"recon.h5", "recon_initial.h5"}
    checkpoint_files = []
    for f in all_files:
        bn = os.path.basename(f)
        if bn in _EXCLUDE or "ending_condition" in bn:
            continue
        if os.path.getmtime(f) < session_start:
            continue
        checkpoint_files.append(f)
    checkpoint_files.sort(key=os.path.getmtime)

    # Also check the main recon.h5 (updated in-place each minibatch)
    main_file = os.path.join(ckpt_dir, "recon.h5")
    main_mtime = 0.0
    if os.path.isfile(main_file) and os.path.getmtime(main_file) >= session_start:
        try:
            with h5py.File(main_file, "r", locking=False) as _probe:
                _ = _probe["densities"].shape
            main_mtime = os.path.getmtime(main_file)
        except Exception:
            pass

    # Pick whichever is freshest: latest checkpoint or recon.h5
    latest_ckpt_mtime = os.path.getmtime(checkpoint_files[-1]) if checkpoint_files else 0.0

    if main_mtime > 0 and main_mtime >= latest_ckpt_mtime:
        # recon.h5 has the most recent data (updated in-place between checkpoints)
        return {"file": main_file, "iteration": -1, "mtime": main_mtime}

    if checkpoint_files:
        latest = checkpoint_files[-1]
        basename = os.path.splitext(os.path.basename(latest))[0]
        parts = basename.rsplit("_", 1)
        try:
            iteration = int(parts[-1])
        except (ValueError, IndexError):
            iteration = -1
        return {"file": latest, "iteration": iteration, "mtime": latest_ckpt_mtime}

    return None


def recon_image_endpoints(session_manager):
    """Register reconstruction image endpoints."""

    @router.get("/get_session_recon_info/")
    def get_session_recon_info(session_id: str = ""):
        """Return metadata about the latest checkpoint for a session."""
        if not session_id:
            return {"status": "no_session"}
        session = session_manager.get_session(session_id)
        if not session:
            return {"status": "no_session"}

        ckpt = _find_latest_checkpoint(session)
        if not ckpt:
            return {"status": "no_checkpoint"}

        filepath = ckpt.get("file", "")
        if not filepath or not os.path.isfile(filepath):
            return {"status": "no_checkpoint"}

        try:
            with h5py.File(filepath, "r", locking=False) as f:
                shape = tuple(f["densities"].shape)  # [n_elem, H, N, N]
                elements = [
                    e.decode("utf-8") if isinstance(e, bytes) else str(e)
                    for e in f["elements"][...]
                ]
            return {
                "status": "ok",
                "iteration": ckpt.get("iteration", -1),
                "mtime": ckpt.get("mtime", 0),
                "elements": elements,
                "n_slices": shape[1],
                "spatial": [shape[2], shape[3]],
                "file": filepath,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @router.get("/get_recon_slice/")
    def get_recon_slice(
        session_id: str = "",
        elem_idx: int = 0,
        slice_idx: int = 0,
        file: str = "",
    ):
        """Read one 2D slice from the latest checkpoint and return as base64 PNG.

        If *file* is provided and exists, skip the checkpoint scan and read
        directly — saves ~3-25 ms of NFS stat calls on interactive requests.
        """
        if not session_id:
            return {"status": "no_session", "data_url": None}
        session = session_manager.get_session(session_id)
        if not session:
            return {"status": "no_session", "data_url": None}

        # Fast path: caller already knows the file from get_session_recon_info.
        filepath = None
        iteration = -1
        if file and os.path.isfile(file):
            filepath = file

        if not filepath:
            ckpt = _find_latest_checkpoint(session)
            if not ckpt:
                return {"status": "no_checkpoint", "data_url": None}
            filepath = ckpt.get("file", "")
            iteration = ckpt.get("iteration", -1)
            if not filepath or not os.path.isfile(filepath):
                return {"status": "no_checkpoint", "data_url": None}

        try:
            result = _read_recon_slice(filepath, elem_idx, slice_idx)
            result["status"] = "ok"
            result["iteration"] = iteration
            return result
        except Exception as e:
            return {"status": "error", "data_url": None, "error": str(e)}
