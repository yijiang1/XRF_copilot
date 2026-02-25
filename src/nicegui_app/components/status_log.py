"""Scrollable monospace status log component."""

from nicegui import ui
from ..state import AppState


def create_status_log(state: AppState) -> tuple[ui.scroll_area, callable]:
    """Create a scrollable monospace log display.

    Returns:
        (log_element, update_log_fn) tuple.
    """
    with ui.row().classes("w-full items-center justify-between mb-1"):
        ui.label("Status:").classes("font-bold")

        def _clear_log():
            state.messages.clear()
            log_container.clear()

        ui.button("Clear", icon="delete_sweep", on_click=_clear_log).props(
            "flat dense size=sm"
        )

    with ui.scroll_area().classes("w-full h-64 border rounded p-2 bg-gray-50") as scroll:
        log_container = ui.column().classes("w-full gap-0")

    prev_msg_count = {"value": 0}

    def update_log():
        current_count = len(state.messages)
        if current_count == prev_msg_count["value"]:
            return

        log_container.clear()
        with log_container:
            for msg in state.messages:
                ui.html(msg).classes("font-mono text-sm whitespace-pre-wrap")

        if current_count > prev_msg_count["value"]:
            ui.run_javascript(f"""
                const scrollArea = document.getElementById('c{scroll.id}');
                if (scrollArea) {{
                    const container = scrollArea.querySelector('.q-scrollarea__container');
                    if (container) {{
                        container.scrollTop = container.scrollHeight;
                    }}
                }}
            """)

        prev_msg_count["value"] = current_count

    return scroll, update_log
