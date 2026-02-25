"""Run/Stop control panel for the reconstruction page."""

import time
from nicegui import ui
from ..api_client import XRFSimulationAPIClient
from ..state import ReconState
from ..utils.message_formatter import append_to_message_list


def _collect_recon_params(input_elements: dict, valid_params: list) -> dict:
    """Gather reconstruction parameter values from UI input elements."""
    params = {}
    for key in valid_params:
        if key not in input_elements:
            continue
        val = input_elements[key].value

        # Integer fields
        if isinstance(val, float) and key in (
            "sample_size_n", "sample_height_n", "gpu_id",
            "n_epochs", "save_every_n_epochs", "minibatch_size",
            "XRT_ratio_dataset_idx",
            "scaler_counts_us_ic_dataset_idx",
            "scaler_counts_ds_ic_dataset_idx",
        ):
            val = int(val)

        params[key] = val
    return params


def create_recon_control_panel(
    state: ReconState,
    api: XRFSimulationAPIClient,
    input_elements: dict,
    valid_params: list,
) -> tuple[ui.button, ui.button]:
    """Create Run/Stop buttons wired to the reconstruction backend API.

    Returns:
        (run_btn, stop_btn) tuple.
    """
    with ui.row().classes("w-full gap-4 mb-4"):
        run_btn = ui.button(
            "Run Reconstruction", icon="play_arrow", color="green"
        ).classes("flex-1")
        stop_btn = ui.button(
            "Stop Reconstruction", icon="stop", color="red"
        ).classes("flex-1")

    async def on_run():
        run_btn.props("color=orange")
        run_btn.disable()
        state.button_status = "processing"
        state.button_timestamp = time.time()

        params = _collect_recon_params(input_elements, valid_params)

        try:
            fresh = []
            fresh = append_to_message_list(
                fresh, "Starting reconstruction process...", level="INFO"
            )
            fresh = append_to_message_list(fresh, "Parameters:", level="INFO")
            for key, value in params.items():
                fresh = append_to_message_list(
                    fresh, f"  {key}: {value}", level="INFO"
                )

            await api.setup_reconstruction(params)
            run_resp = await api.run_reconstruction()
            status_msg = run_resp.get("status", "Reconstruction started successfully.")
            fresh = append_to_message_list(fresh, status_msg, level="WORKER")

            state.messages = fresh
            state.results_ready = False

        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error starting reconstruction: {e}", level="ERROR"
            )
            run_btn.props("color=green")
            run_btn.enable()
            state.button_status = "idle"

    async def on_stop():
        state.messages = append_to_message_list(
            state.messages, "Stopping reconstruction...", level="WARNING"
        )
        try:
            resp = await api.stop_reconstruction()
            status = resp.get("status", "Reconstruction stopped.")
            state.messages = append_to_message_list(
                state.messages, status, level="SUCCESS"
            )
        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error stopping reconstruction: {e}", level="ERROR"
            )

        run_btn.props("color=green")
        run_btn.enable()
        state.button_status = "idle"
        state.is_busy = False

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)

    return run_btn, stop_btn
