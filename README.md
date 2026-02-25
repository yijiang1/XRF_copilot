> **⚠️ Important**
> This package is actively in development and breaking changes may occur.

# XRF Copilot

XRF Copilot is a web-based GUI for X-ray fluorescence (XRF) data analysis at the Advanced Photon Source (APS/ANL). It provides three integrated workflows:

| Workflow | Description |
|---|---|
| **XRF Simulation** | Simulate XRF maps (with optional self-absorption / probe attenuation) via a GPU-accelerated forward model |
| **XRF Tomographic Reconstruction** | Reconstruct 3D element distributions from XRF tomography data |
| **FL Self-Absorption Correction** | Iterative fluorescence self-absorption correction based on Ge et al., *Commun. Mater.* **3**, 37 (2022), with interactive HDF5 data inspection |

## Architecture

The application is split into a **backend** (FastAPI, runs on the GPU machine) and a **frontend** (NiceGUI, accessed from any browser). They communicate over HTTP, with an auto-generated API key protecting the backend.

```
Browser → NiceGUI frontend (port 8050) → FastAPI backend (port 8000) → Worker process (GPU)
```

The **launcher** (`launcher.py`) handles starting both components — it SSH's to the remote GPU machine to start the backend, then starts the frontend locally and opens it in the browser.

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/yijiang1/XRF_copilot.git
cd XRF_copilot
```

### 2. Create a conda environment (Python 3.12)

```bash
conda create --name xrf_copilot python=3.12
conda activate xrf_copilot
```

### 3. Install GPU / scientific packages via conda

These packages require conda for correct CUDA linking:

```bash
# ASTRA tomography toolbox (GPU-accelerated)
conda install -c astra-toolbox astra-toolbox

# tomopy (sinogram pre-processing)
conda install -c conda-forge tomopy

# Numba (JIT + CUDA kernels) — pin to 0.63.x for compatibility
conda install numba=0.63

# CUDA compiler (for numba CUDA kernels) — match your driver
conda install -c conda-forge cuda-nvcc=12.6

# xraylib (fluorescence cross-sections and atomic data)
conda install -c conda-forge xraylib
```

### 4. Install Python packages

```bash
pip install -r requirements.txt
```

Or install as a package (makes `xrf-gui` and `xrf-backend` CLI commands available):

```bash
pip install -e .
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `HOST` | Frontend host (default `0.0.0.0`) |
| `PORT` | Frontend port (default `8050`) |
| `API_ENDPOINT` | URL of the FastAPI backend, e.g. `http://gpu-machine:8000` |
| `BACKEND_API_KEY` | Shared secret between frontend and backend |
| `ANL_USERNAME` | Argonne username for Argo Gateway (LLM chat assistant) |
| `ARGO_BASE_URL` | Argo Gateway base URL |
| `ARGO_MODEL` | Model to use, e.g. `gpt4o` |

## Running the Application

### Option A — Launcher (recommended)

The launcher handles SSH authentication, backend startup, and frontend startup with a single click.

```bash
conda activate xrf_copilot
python launcher.py
```

A browser window opens at `http://localhost:8060/`. Fill in the remote machine hostname, your ANL username and password, and click **Launch**. The launcher will:

1. SSH to the remote GPU machine and start the FastAPI backend
2. Start the NiceGUI frontend locally
3. Open the app URL in your browser

> **Note**: The launcher supports both direct SSH (paramiko) and a two-step `su - user → ssh host` path, which is needed on APS beamline machines where direct SSH from user accounts is restricted.

### Option B — Manual startup

**On the GPU machine (backend):**

```bash
conda activate xrf_copilot
cd /path/to/XRF_copilot
python -m src.services.main --host 0.0.0.0 --port 8000 --api-key <your-key>
```

**On any machine with a browser (frontend):**

```bash
conda activate xrf_copilot
cd /path/to/XRF_copilot
export API_ENDPOINT=http://<gpu-machine>:8000
export BACKEND_API_KEY=<your-key>
python -m src.nicegui_app --api-key <your-key>
```

Then open `http://<frontend-host>:8050/<your-key>` in your browser.

## Features

### HDF5 Data Inspector (FL Correction page)

- Slice-based viewing of large `everything.h5` files — only loads the displayed slice (~1.4 MB), never the full array
- **Zoom/pan** with Ctrl+wheel; double-click to reset
- **Ruler tool** — draw a line to measure distances; shows physical distance in nm or µm using the configured voxel size
- **Crop tool** — draw a symmetric crop box about the rotation axis; box persists and resizes correctly when the browser window is resized
- Channel and angle selectors with debounced slider

### FL Parameter Form

- Periodic-table tile selector for element channels (auto-populated from the HDF5 file)
- Per-element K/L shell toggle with xraylib auto-fill for density and emission energy
- Ion-chamber channel selector
- Crop region set interactively via the data inspector (no manual pixel input required)

### Chat Assistant ("Fluoro")

An LLM-powered assistant (Argo Gateway) that can help specify simulation and reconstruction parameters in natural language. Supports free-form description, guided inquiry, and domain-rule-based suggestion modes.

## Known CUDA Compatibility Notes

The `xrf_copilot` environment uses numba 0.63 with CUDA. If the system CUDA driver does not support the PTX version generated by the installed `cuda-nvcc`, a library swap may be needed:

- **PTX version mismatch**: If `cuda-nvcc 12.9` generates PTX 8.8/8.9 but the driver only accepts PTX ≤ 8.7, swap `libnvvm.so.4.0.0` with the CUDA 12.6 version from another conda package cache.
- **Cooperative launch limit**: On A100 GPUs (108 SMs), the default `blocks=256` in the MLEM CUDA kernel exceeds the cooperative launch limit. The bundled `src/fl_correction/numba_util.py` auto-detects the SM count and sets `blocks = min(2 × sm_count, 216)`.

## Project Structure

```
XRF_copilot/
├── launcher.py                   # GUI launcher (SSH backend + local frontend)
├── requirements.txt              # pip dependencies
├── pyproject.toml                # package config
├── .env.example                  # environment variable template
│
└── src/
    ├── nicegui_app/              # NiceGUI frontend
    │   ├── main.py               # Entry point (ui.run)
    │   ├── config.py             # .env loading
    │   ├── state.py              # App state dataclasses
    │   ├── api_client.py         # Async httpx client to backend
    │   ├── pages/                # One module per page (simulation, reconstruction, fl_correction)
    │   ├── components/           # Reusable UI components
    │   │   ├── h5_inspector.py   # Interactive HDF5 slice viewer
    │   │   ├── fl_parameter_form.py  # FL correction parameter form
    │   │   └── ...
    │   └── services/             # LLM chat assistant + knowledge files
    │
    ├── services/                 # FastAPI backend
    │   ├── main.py               # FastAPI app + API key middleware
    │   ├── models.py             # Pydantic parameter models
    │   ├── worker.py             # Simulation worker (multiprocessing)
    │   ├── fl_worker.py          # FL correction worker (multiprocessing)
    │   ├── recon_worker.py       # Reconstruction worker (multiprocessing)
    │   └── routes/               # One module per endpoint group
    │
    ├── simulation/               # XRF forward simulation engine
    ├── fl_correction/            # BNL FL self-absorption correction engine
    └── reconstruction/           # XRF tomographic reconstruction engine
```

## References

| Tool / Algorithm | Source |
|---|---|
| XRF simulation forward model | [hpphappy/XRF_tomography](https://github.com/hpphappy/XRF_tomography) — see also [Thesis](https://web.archive.org/web/20220923101648/https://arch.library.northwestern.edu/downloads/707958231) |
| FL self-absorption correction | Ge, M., Huang, X., Yan, H. et al., "Three-dimensional imaging of grain boundaries via quantitative fluorescence X-ray tomography analysis," *Commun. Mater.* **3**, 37 (2022). [https://doi.org/10.1038/s43246-022-00259-x](https://doi.org/10.1038/s43246-022-00259-x) |
| ASTRA tomography toolbox | [astra-toolbox.com](https://www.astra-toolbox.com/) |
| Argo Gateway (LLM) | Argonne internal API — requires ANL credentials |
