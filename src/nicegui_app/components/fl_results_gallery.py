"""FL correction results gallery: per-element 2D slice heatmaps.

Loads the latest recon_XX.h5 from the recon directory.
HDF5 structure: f[element_name] → (n_slices, ny, nx) float32 array.
"""

import os
import glob
import h5py
import numpy as np
import plotly.graph_objects as go
from nicegui import ui, run
from ..state import FLState


def _load_fl_results(recon_dir: str) -> tuple[list, list, str]:
    """Load the latest FL correction HDF5 file and return Plotly figures.

    Runs in a thread pool via run.io_bound.

    Returns:
        (figures, elements, filename) tuple.
    """
    pattern = os.path.join(recon_dir, "recon_[0-9]*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No recon_*.h5 files found in {recon_dir}")

    latest_file = files[-1]
    figures = []
    elements = []

    with h5py.File(latest_file, "r") as f:
        for elem in f.keys():
            data = np.array(f[elem])  # (n_slices, ny, nx)
            mid = data.shape[0] // 2
            slice_data = data[mid]

            fig = go.Figure(
                data=go.Heatmap(z=slice_data, colorscale="Viridis", showscale=True)
            )
            fig.update_layout(
                title=f"{elem} (slice {mid}/{data.shape[0]})",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
            )
            figures.append(fig)
            elements.append(elem)

    return figures, elements, os.path.basename(latest_file)


def create_fl_results_gallery(state: FLState) -> callable:
    """Create the FL correction results display area with Plotly heatmaps.

    Returns an async update_results() callable to be called from the polling loop.
    """
    ui.label("Results:").classes("font-bold mt-4 mb-1")
    results_container = ui.column().classes("w-full gap-2")
    with results_container:
        ui.label("Run FL correction to see results here.").classes(
            "text-gray-400 text-center w-full py-8"
        )

    async def update_results():
        if not state.results_ready or state.results_displayed:
            return
        if not state.recon_file:
            return

        state.is_busy = True
        try:
            figures, elements, fname = await run.io_bound(
                _load_fl_results, state.recon_file
            )
            results_container.clear()
            with results_container:
                ui.label(f"Results from: {fname}").classes(
                    "text-green-600 font-mono text-sm mb-1"
                )
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    for fig in figures:
                        with ui.card().classes("flex-1 min-w-[300px]"):
                            ui.plotly(fig).classes("w-full")
            state.results_displayed = True
        except Exception as e:
            results_container.clear()
            with results_container:
                ui.label(f"Error loading FL results: {e}").classes("text-red-500")
        finally:
            state.is_busy = False

    return update_results
