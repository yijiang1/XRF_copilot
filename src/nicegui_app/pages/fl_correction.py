"""FL self-absorption correction page (BNL)."""

import time
import httpx
from urllib.parse import urlparse
from nicegui import ui
from ..state import FLState
from ..api_client import XRFSimulationAPIClient
from ..components.fl_parameter_form import create_fl_parameter_form
from ..components.fl_control_panel import create_fl_control_panel
from ..components.status_log import create_status_log
from ..components.fl_results_gallery import create_fl_results_gallery
from ..utils.message_formatter import append_to_message_list


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 8000)
    return host, port


def create_fl_correction_page(api_key: str = ""):
    """Build the FL correction page."""
    state = FLState()
    api = XRFSimulationAPIClient(api_key=api_key)

    backend_connected = {"value": False, "error_logged": False}
    init_host, init_port = _parse_endpoint(api.endpoint)

    # Backend connection controls
    with ui.row().classes("w-full justify-center items-end gap-2 mb-2 px-4"):
        ui.label("Backend:").classes("text-sm font-bold self-center")
        host_input = ui.input("Host", value=init_host).classes("w-64").props("dense outlined")
        port_input = ui.input("Port", value=init_port).classes("w-24").props("dense outlined")
        api_key_input = ui.input(
            "API Key", value=api_key, password=True, password_toggle_button=True
        ).classes("w-48").props("dense outlined")
        connect_btn = ui.button("Connect", icon="link").props("dense")

    connection_label = ui.label(
        f"Backend: {api.endpoint}"
    ).classes("text-sm text-gray-400 text-center w-full mb-2")

    def on_connect():
        host = host_input.value.strip()
        port = port_input.value.strip()
        new_endpoint = f"http://{host}:{port}"
        api.set_endpoint(new_endpoint)
        api.set_api_key(api_key_input.value.strip())
        backend_connected["value"] = False
        backend_connected["error_logged"] = False
        connection_label.set_text(f"Backend: {new_endpoint} (connecting...)")
        connection_label.classes(
            remove="text-green-600 text-red-500 text-orange-500",
            add="text-gray-400",
        )

    connect_btn.on_click(on_connect)

    with ui.column().classes("w-full px-4 gap-4"):
        input_elements, valid_params = create_fl_parameter_form(state)
        run_btn, stop_btn = create_fl_control_panel(state, api, input_elements, valid_params)
        _log_area, update_log = create_status_log(state)

        # Step-based progress display
        ui.label("Progress:").classes("font-bold mt-4 mb-1")
        step_info = ui.label("No correction running").classes(
            "text-gray-500 text-xl font-semibold w-full mb-1"
        )
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("mb-2")
        progress_label = ui.label("0%").classes("text-center text-sm")

        def update_progress():
            pct = state.progress_percent
            progress_bar.set_value(pct / 100.0)
            progress_label.set_text(f"{pct:.1f}%")
            if state.is_running or state.current_step > 0:
                label = state.step_label or f"Step {state.current_step}/{state.total_steps}"
                step_info.set_text(f"{label}  ({pct:.0f}%)")
            else:
                step_info.set_text("No correction running")

        update_results = create_fl_results_gallery(state)

    error_reported = {"value": False}
    worker_log_offset = {"value": 0}

    async def poll_backend():
        try:
            # Progress
            progress_data = await api.get_fl_progress()
            state.is_running = progress_data.get("is_running", False)
            state.current_step = progress_data.get("current_step", 0)
            state.total_steps = progress_data.get("total_steps", 0)
            state.step_label = progress_data.get("step_label", "")
            state.progress_percent = progress_data.get("progress_percent", 0.0)

            # Errors
            try:
                fl_status = await api.get_fl_status()
                fl_error = fl_status.get("error")
                if fl_error and not error_reported["value"]:
                    error_reported["value"] = True
                    state.messages = append_to_message_list(
                        state.messages, f"FL correction error: {fl_error}", level="ERROR"
                    )
                    ui.notify(
                        "FL correction failed. See status log.",
                        type="negative", position="top", timeout=10000,
                    )
                    run_btn.props("color=green")
                    run_btn.enable()
                    state.button_status = "idle"
                    state.is_busy = False
                elif not fl_error:
                    error_reported["value"] = False
            except Exception:
                pass

            # Results
            try:
                results_data = await api.get_fl_results()
                if results_data.get("results_ready") and not state.results_ready:
                    state.recon_file = results_data.get("recon_file", "")
                    state.results_ready = True
                    ui.notify(
                        "FL correction complete!",
                        type="positive", position="top", timeout=5000,
                    )
            except Exception:
                pass

            # Worker logs
            if state.is_running:
                try:
                    worker_data = await api.get_fl_worker_status()
                    worker_logs = worker_data.get("worker_logs", [])
                    if len(worker_logs) < worker_log_offset["value"]:
                        worker_log_offset["value"] = 0
                    for log_entry in worker_logs[worker_log_offset["value"]:]:
                        if isinstance(log_entry, dict):
                            msg = log_entry.get("message", str(log_entry))
                            level = log_entry.get("level", "WORKER")
                        else:
                            msg = str(log_entry)
                            level = "WORKER"
                        state.messages = append_to_message_list(state.messages, msg, level=level)
                    worker_log_offset["value"] = len(worker_logs)
                except Exception:
                    pass

            # Button state
            if state.is_running:
                run_btn.props("color=orange")
                run_btn.disable()
            elif state.button_status == "processing":
                if time.time() - state.button_timestamp > 10:
                    run_btn.props("color=green")
                    run_btn.enable()
                    state.button_status = "idle"
            else:
                run_btn.props("color=green")
                run_btn.enable()

            if not backend_connected["value"]:
                backend_connected["value"] = True
                backend_connected["error_logged"] = False
                connection_label.set_text(f"Backend: {api.endpoint} (connected)")
                connection_label.classes(
                    remove="text-red-500 text-gray-400", add="text-green-600"
                )

        except httpx.HTTPStatusError as e:
            if not backend_connected["error_logged"]:
                backend_connected["error_logged"] = True
                backend_connected["value"] = False
                label = "(invalid API key)" if e.response.status_code == 403 else f"(error {e.response.status_code})"
                connection_label.set_text(f"Backend: {api.endpoint} {label}")
                connection_label.classes(
                    remove="text-green-600 text-gray-400",
                    add="text-orange-500" if e.response.status_code == 403 else "text-red-500",
                )
        except Exception:
            if not backend_connected["error_logged"]:
                backend_connected["error_logged"] = True
                backend_connected["value"] = False
                connection_label.set_text(f"Backend: {api.endpoint} (not reachable)")
                connection_label.classes(
                    remove="text-green-600 text-gray-400 text-orange-500",
                    add="text-red-500",
                )

        update_log()
        update_progress()
        await update_results()

    ui.timer(2.0, poll_backend)
