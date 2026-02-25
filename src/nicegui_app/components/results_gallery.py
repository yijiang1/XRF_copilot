"""XRF/XRT results gallery with Plotly heatmaps."""

import h5py
import plotly.graph_objects as go
from nicegui import ui, run
from ..state import AppState


def _create_empty_figure(title: str) -> go.Figure:
    """Create a placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text="No data", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="gray"))
        ],
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def _load_results(sim_xrf_file: str, sim_xrt_file: str) -> tuple[list, list, object]:
    """Load XRF and XRT data from HDF5 files (runs in thread pool)."""
    xrf_figures = []
    elements = []

    with h5py.File(sim_xrf_file, "r") as f:
        xrf_data = f["exchange/data"][:]
        raw_elements = f["exchange/elements"][:]
        elements = [el.decode("utf-8") if isinstance(el, bytes) else str(el) for el in raw_elements]

    for i, element in enumerate(elements):
        fig = go.Figure(
            data=go.Heatmap(z=xrf_data[i], colorscale="Viridis", showscale=True)
        )
        fig.update_layout(
            title=f"XRF: {element}",
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
        )
        xrf_figures.append(fig)

    # Load XRT
    with h5py.File(sim_xrt_file, "r") as f:
        xrt_data = f["exchange/data"][:]

    xrt_fig = go.Figure(
        data=go.Heatmap(z=xrt_data[-1], colorscale="gray", showscale=True)
    )
    xrt_fig.update_layout(
        title="XRT Transmission",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )

    return xrf_figures, elements, xrt_fig


def create_results_gallery(state: AppState) -> tuple[dict, callable]:
    """Create the results display area with Plotly heatmaps.

    Returns:
        (elements_dict, update_results_fn) tuple.
    """
    ui.label("Results:").classes("font-bold mt-4 mb-1")

    results_container = ui.column().classes("w-full gap-2")

    # Show placeholder
    with results_container:
        placeholder_label = ui.label("Run a simulation to see results here.").classes(
            "text-gray-400 text-center w-full py-8"
        )

    gallery_elements = {"container": results_container, "placeholder": placeholder_label}

    async def update_results():
        """Check for new results and update the display."""
        if not state.results_ready or state.results_displayed:
            return

        if not state.sim_xrf_file or not state.sim_xrt_file:
            return

        state.is_busy = True
        try:
            xrf_figures, elements, xrt_fig = await run.io_bound(
                _load_results, state.sim_xrf_file, state.sim_xrt_file
            )

            results_container.clear()
            with results_container:
                # XRF maps in a grid
                n_cols = min(len(xrf_figures), 3)
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    for fig in xrf_figures:
                        with ui.card().classes("flex-1 min-w-[300px]"):
                            ui.plotly(fig).classes("w-full")

                # XRT map full width
                with ui.card().classes("w-full"):
                    ui.plotly(xrt_fig).classes("w-full")

            state.results_displayed = True

        except Exception as e:
            results_container.clear()
            with results_container:
                ui.label(f"Error loading results: {e}").classes("text-red-500")
        finally:
            state.is_busy = False

    return gallery_elements, update_results
