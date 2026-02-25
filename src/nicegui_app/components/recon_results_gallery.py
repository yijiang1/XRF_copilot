"""Reconstruction results gallery: per-element 2D slice heatmaps.

Loads from recon_grid.h5:
  s["sample/densities"] → (n_elem, height, ny, nx) float32
  s["sample/elements"]  → (n_elem,) byte strings
"""

import h5py
import numpy as np
import plotly.graph_objects as go
from nicegui import ui, run
from ..state import ReconState


def _load_recon_results(recon_file: str) -> tuple[list, list]:
    """Load reconstruction HDF5 and return Plotly figures and element names.

    Runs in a thread pool via run.io_bound.

    Returns:
        (figures, elements) tuple.
    """
    with h5py.File(recon_file, "r") as f:
        densities = np.array(f["sample/densities"])  # (n_elem, height, ny, nx)
        raw_elems = np.array(f["sample/elements"])
        elements = [
            e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in raw_elems
        ]

    figures = []
    for i, elem in enumerate(elements):
        vol = densities[i]  # (height, ny, nx)
        mid = vol.shape[0] // 2
        slice_data = vol[mid]

        fig = go.Figure(
            data=go.Heatmap(z=slice_data, colorscale="Viridis", showscale=True)
        )
        fig.update_layout(
            title=f"{elem} (height slice {mid}/{vol.shape[0]})",
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
        )
        figures.append(fig)

    return figures, elements


def create_recon_results_gallery(state: ReconState) -> callable:
    """Create the reconstruction results display area with Plotly heatmaps.

    Returns an async update_results() callable to be called from the polling loop.
    """
    ui.label("Results:").classes("font-bold mt-4 mb-1")
    results_container = ui.column().classes("w-full gap-2")
    with results_container:
        ui.label("Run reconstruction to see results here.").classes(
            "text-gray-400 text-center w-full py-8"
        )

    async def update_results():
        if not state.results_ready or state.results_displayed:
            return
        if not state.recon_file:
            return

        state.is_busy = True
        try:
            figures, elements = await run.io_bound(
                _load_recon_results, state.recon_file
            )
            results_container.clear()
            with results_container:
                ui.label(f"File: {state.recon_file}").classes(
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
                ui.label(f"Error loading reconstruction results: {e}").classes(
                    "text-red-500"
                )
        finally:
            state.is_busy = False

    return update_results
