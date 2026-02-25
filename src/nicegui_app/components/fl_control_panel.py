"""Run/Stop control panel for the FL correction page (BNL)."""

import time
from nicegui import ui
from ..api_client import XRFSimulationAPIClient
from ..state import FLState
from ..utils.message_formatter import append_to_message_list


def _collect_fl_params(input_elements: dict, valid_params: list) -> dict:
    """Gather FL correction parameter values from UI input elements."""
    params = {}
    int_fields = {
        # ic_channel_idx and crop_x_{start,end} are no longer ui.number widgets;
        # they come in as int or are absent — no float→int conversion needed.
        "binning_factor", "mask_length_maximum",
        "recon_n_iter", "n_correction_iters", "correction_n_iter",
        "border_pixels", "smooth_filter_size", "num_cpu",
    }
    for key in valid_params:
        if key not in input_elements:
            continue
        val = input_elements[key].value
        if isinstance(val, float) and key in int_fields:
            val = int(val)
        params[key] = val
    return params


def create_fl_control_panel(
    state: FLState,
    api: XRFSimulationAPIClient,
    input_elements: dict,
    valid_params: list,
) -> tuple[ui.button, ui.button]:
    """Create Run/Stop buttons wired to the FL correction backend API.

    Returns:
        (run_btn, stop_btn) tuple.
    """
    with ui.row().classes("w-full gap-4 mb-4"):
        run_btn = ui.button(
            "Run FL Correction", icon="play_arrow", color="green"
        ).classes("flex-1")
        stop_btn = ui.button(
            "Stop", icon="stop", color="red"
        ).classes("flex-1")

    async def on_run():
        run_btn.props("color=orange")
        run_btn.disable()
        state.button_status = "processing"
        state.button_timestamp = time.time()

        params = _collect_fl_params(input_elements, valid_params)

        try:
            fresh = []
            fresh = append_to_message_list(fresh, "Starting FL correction...", level="INFO")
            fresh = append_to_message_list(fresh, f"  fn_root: {params.get('fn_root', '')}", level="INFO")
            fresh = append_to_message_list(fresh, f"  n_correction_iters: {params.get('n_correction_iters', 4)}", level="INFO")

            await api.setup_fl_correction(params)
            run_resp = await api.run_fl_correction()
            status_msg = run_resp.get("status", "FL correction started.")
            fresh = append_to_message_list(fresh, status_msg, level="WORKER")

            state.messages = fresh
            state.results_ready = False
            state.recon_file = ""
            state.current_step = 0
            state.total_steps = 0
            state.step_label = ""

        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error starting FL correction: {e}", level="ERROR"
            )
            run_btn.props("color=green")
            run_btn.enable()
            state.button_status = "idle"

    async def on_stop():
        state.messages = append_to_message_list(
            state.messages, "Stopping FL correction...", level="WARNING"
        )
        try:
            resp = await api.stop_fl_correction()
            state.messages = append_to_message_list(
                state.messages, resp.get("status", "Stopped."), level="SUCCESS"
            )
        except Exception as e:
            state.messages = append_to_message_list(
                state.messages, f"Error stopping: {e}", level="ERROR"
            )

        run_btn.props("color=green")
        run_btn.enable()
        state.button_status = "idle"
        state.is_busy = False

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)

    return run_btn, stop_btn
