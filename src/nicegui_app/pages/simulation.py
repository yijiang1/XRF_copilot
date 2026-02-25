"""Main simulation page — composes all components and sets up polling."""

import time
import httpx
from urllib.parse import urlparse
from nicegui import ui
from ..state import AppState
from ..api_client import XRFSimulationAPIClient
from ..components.parameter_form import create_parameter_form
from ..components.control_panel import create_control_panel
from ..components.status_log import create_status_log
from ..components.progress_display import create_progress_display
from ..components.results_gallery import create_results_gallery
from ..components.chat_assistant import create_chat_assistant
from ..utils.message_formatter import append_to_message_list


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    """Extract host and port from an endpoint URL."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 8000)
    return host, port


def create_simulation_page(api_key: str = ""):
    """Build the main simulation page.

    Composes all components into a single-column layout and sets up
    a polling timer that updates every 2 seconds.
    """
    state = AppState()
    api = XRFSimulationAPIClient(api_key=api_key)

    backend_connected = {"value": False, "error_logged": False}

    init_host, init_port = _parse_endpoint(api.endpoint)

    # Backend connection controls
    with ui.row().classes("w-full justify-center items-end gap-2 mb-2 px-4"):
        ui.label("Backend:").classes("text-sm font-bold self-center")
        host_input = ui.input("Host", value=init_host).classes("w-64").props(
            "dense outlined"
        )
        port_input = ui.input("Port", value=init_port).classes("w-24").props(
            "dense outlined"
        )
        api_key_input = ui.input(
            "API Key", value=api_key, password=True, password_toggle_button=True
        ).classes("w-48").props("dense outlined")
        connect_btn = ui.button("Connect", icon="link").props("dense")

    # Connection status
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
        # Mutable ref so chat assistant can access input_elements after form is built
        elements_ref = {"elements": {}, "valid": []}

        def _inject_chat():
            create_chat_assistant(state, elements_ref)

        # Parameter form (chat injected before expansion sections)
        input_elements, valid_params = create_parameter_form(
            state, pre_sections_callback=_inject_chat
        )
        elements_ref["elements"] = input_elements
        elements_ref["valid"] = valid_params

        # Controls
        run_btn, stop_btn = create_control_panel(
            state, api, input_elements, valid_params
        )
        _log_area, update_log = create_status_log(state)
        _progress_bar, update_progress = create_progress_display(state)
        _gallery_elements, update_results = create_results_gallery(state)

    # Track error reporting
    error_reported = {"value": False}
    worker_log_offset = {"value": 0}

    # --- Timer-based polling ---
    async def poll_backend():
        """Called every 2 seconds to poll the backend for updates."""
        try:
            # Get progress data
            progress_data = await api.get_progress()
            state.progress_percent = progress_data.get("progress_percent", 0)
            state.is_running = progress_data.get("is_running", False)
            state.current_batch = progress_data.get("current_batch", 0)
            state.total_batches = progress_data.get("total_batches", 0)

            # Check for errors
            try:
                sim_status = await api.get_simulation_status()
                sim_error = sim_status.get("error")
                if sim_error and not error_reported["value"]:
                    error_reported["value"] = True
                    state.messages = append_to_message_list(
                        state.messages,
                        f"Simulation error: {sim_error}",
                        level="ERROR",
                    )
                    ui.notify(
                        "Simulation failed. See status log for details.",
                        type="negative",
                        position="top",
                        timeout=10000,
                    )
                    run_btn.props("color=green")
                    run_btn.enable()
                    state.button_status = "idle"
                    state.is_busy = False
                elif not sim_error:
                    error_reported["value"] = False
            except Exception:
                pass

            # Check for results
            try:
                results_data = await api.get_results()
                if results_data.get("results_ready"):
                    state.sim_xrf_file = results_data.get("sim_xrf_file", "")
                    state.sim_xrt_file = results_data.get("sim_xrt_file", "")
                    state.results_ready = True
            except Exception:
                pass

            # Get worker logs while running
            if state.is_running:
                try:
                    worker_data = await api.get_worker_status()
                    worker_logs = worker_data.get("worker_logs", [])

                    if len(worker_logs) < worker_log_offset["value"]:
                        worker_log_offset["value"] = 0

                    new_logs = worker_logs[worker_log_offset["value"]:]
                    for log_entry in new_logs:
                        if isinstance(log_entry, dict):
                            msg = log_entry.get("message", str(log_entry))
                            level = log_entry.get("level", "WORKER")
                        else:
                            msg = str(log_entry)
                            level = "WORKER"

                        state.messages = append_to_message_list(
                            state.messages, msg, level=level
                        )
                    worker_log_offset["value"] = len(worker_logs)
                except Exception:
                    pass

            # Manage button state
            if state.is_running:
                run_btn.props("color=orange")
                run_btn.disable()
            elif state.button_status == "processing":
                time_diff = time.time() - state.button_timestamp
                if time_diff > 10:
                    run_btn.props("color=green")
                    run_btn.enable()
                    state.button_status = "idle"
            else:
                run_btn.props("color=green")
                run_btn.enable()

            # Mark backend as connected
            if not backend_connected["value"]:
                backend_connected["value"] = True
                backend_connected["error_logged"] = False
                connection_label.set_text(
                    f"Backend: {api.endpoint} (connected)"
                )
                connection_label.classes(
                    remove="text-red-500 text-gray-400", add="text-green-600"
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                if not backend_connected["error_logged"]:
                    backend_connected["error_logged"] = True
                    backend_connected["value"] = False
                    connection_label.set_text(
                        f"Backend: {api.endpoint} (invalid API key)"
                    )
                    connection_label.classes(
                        remove="text-green-600 text-gray-400 text-red-500",
                        add="text-orange-500",
                    )
            else:
                if not backend_connected["error_logged"]:
                    backend_connected["error_logged"] = True
                    backend_connected["value"] = False
                    connection_label.set_text(
                        f"Backend: {api.endpoint} (error {e.response.status_code})"
                    )
                    connection_label.classes(
                        remove="text-green-600 text-gray-400 text-orange-500",
                        add="text-red-500",
                    )

        except Exception:
            if not backend_connected["error_logged"]:
                backend_connected["error_logged"] = True
                backend_connected["value"] = False
                connection_label.set_text(
                    f"Backend: {api.endpoint} (not reachable)"
                )
                connection_label.classes(
                    remove="text-green-600 text-gray-400 text-orange-500",
                    add="text-red-500",
                )

        # Update all UI components
        update_log()
        update_progress()
        await update_results()

    ui.timer(2.0, poll_backend)
