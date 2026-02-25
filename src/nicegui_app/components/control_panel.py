"""Run/Stop control panel component."""

import time
from nicegui import ui
from ..api_client import XRFSimulationAPIClient
from ..state import AppState
from ..utils.message_formatter import append_to_message_list


def _collect_params(input_elements: dict, valid_params: list) -> dict:
    """Gather parameter values from the UI input elements."""
    params = {}
    for key in valid_params:
        if key not in input_elements:
            continue
        val = input_elements[key].value

        # Normalize floats that are whole numbers to int
        if isinstance(val, float) and val == int(val) and key in ("gpu_id",):
            val = int(val)

        # Parse elements: comma-separated string -> list of strings
        if key == "elements" and isinstance(val, str):
            val = [e.strip() for e in val.split(",") if e.strip()]

        # Parse rotation_angles: comma-separated string -> list of floats
        if key == "rotation_angles" and isinstance(val, str):
            try:
                val = [float(x.strip()) for x in val.split(",") if x.strip()]
            except ValueError:
                val = [0.0, 0.0, 0.0]

        params[key] = val
    return params


def create_control_panel(
    state: AppState,
    api: XRFSimulationAPIClient,
    input_elements: dict,
    valid_params: list,
) -> tuple[ui.button, ui.button]:
    """Create Run/Stop buttons wired to the backend API.

    Returns:
        (run_btn, stop_btn) tuple.
    """
    with ui.row().classes("w-full gap-4 mb-4"):
        run_btn = ui.button(
            "Run Simulation", icon="play_arrow", color="green"
        ).classes("flex-1")
        stop_btn = ui.button(
            "Stop Simulation", icon="stop", color="red"
        ).classes("flex-1")

    async def on_run():
        run_btn.props("color=orange")
        run_btn.disable()
        state.button_status = "processing"
        state.button_timestamp = time.time()

        params = _collect_params(input_elements, valid_params)

        try:
            fresh = []
            fresh = append_to_message_list(
                fresh, "Starting simulation process...", level="INFO"
            )
            fresh = append_to_message_list(
                fresh, "Parameters:", level="INFO"
            )
            for key, value in params.items():
                fresh = append_to_message_list(
                    fresh, f"  {key}: {value}", level="INFO"
                )

            await api.setup_simulation(params)
            run_resp = await api.run_simulation()
            status_msg = run_resp.get(
                "status", "Simulation started successfully."
            )
            fresh = append_to_message_list(fresh, status_msg, level="WORKER")

            state.messages = fresh
            state.results_ready = False
            state.results_displayed = False

        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error starting simulation: {e}", level="ERROR"
            )
            run_btn.props("color=green")
            run_btn.enable()
            state.button_status = "idle"

    async def on_stop():
        state.messages = append_to_message_list(
            state.messages, "Stopping simulation...", level="WARNING"
        )
        try:
            resp = await api.stop_simulation()
            status = resp.get("status", "Simulation stopped.")
            state.messages = append_to_message_list(
                state.messages, status, level="SUCCESS"
            )
        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error stopping simulation: {e}", level="ERROR"
            )

        run_btn.props("color=green")
        run_btn.enable()
        state.button_status = "idle"
        state.is_busy = False

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)

    return run_btn, stop_btn
