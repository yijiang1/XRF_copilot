"""Horizontal session bar showing compact chips for each reconstruction session."""

from typing import Callable
from nicegui import ui


# Status → (Quasar color name, icon)
_STATUS_STYLE = {
    "running":   ("amber-8",  "play_arrow"),
    "completed": ("green",    "check_circle"),
    "error":     ("red",      "error"),
    "stopped":   ("grey",     "stop_circle"),
}


def create_session_bar(
    on_select: Callable[[str], None],
    on_remove: Callable[[str], None],
    on_clear_finished: Callable[[], None] | None = None,
) -> tuple[ui.element, Callable]:
    """Create a horizontal bar of session chips.

    Args:
        on_select: Called with session_id when a chip is clicked.
        on_remove: Called with session_id when the X button is clicked.
        on_clear_finished: Called when the "Clear Finished" button is clicked.

    Returns:
        (container, update_fn) where update_fn(sessions_list, selected_id)
        refreshes the display.
    """
    container = ui.column().classes("w-full")
    with container:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("list", size="xs").classes("text-gray-400")
                ui.label("Sessions").classes("section-header").style("margin-bottom: 0;")
            if on_clear_finished:
                ui.button(
                    "Clear Finished", icon="delete_outline",
                    on_click=lambda: on_clear_finished(),
                ).props("flat dense no-caps size=sm color=grey")
        chip_row = ui.row().classes("w-full gap-2 flex-wrap items-center")

    def update(sessions: list[dict], selected_id: str):
        """Rebuild session chips from the session list."""
        chip_row.clear()
        if not sessions:
            with chip_row:
                ui.label("No sessions").classes("text-xs text-gray-400 italic")
            return

        with chip_row:
            for s in sessions:
                sid = s["session_id"]
                is_running = s.get("is_running", False)
                has_error  = s.get("error") is not None and not is_running
                is_selected = sid == selected_id

                if is_running:
                    status = "running"
                elif has_error:
                    status = "error"
                else:
                    status = "completed"

                color, icon = _STATUS_STYLE.get(status, ("grey", "help"))
                progress = s.get("progress_percent", 0)
                name = s.get("display_name", sid[:8])

                # Build chip label: "BNL GPU:2 | 42%"
                label_parts = [name]
                if is_running:
                    label_parts.append(f"{progress:.0f}%")
                label = "  |  ".join(label_parts)

                with ui.row().classes("items-center gap-0"):
                    btn = ui.button(
                        label, icon=icon,
                        on_click=lambda _, _sid=sid: on_select(_sid),
                    ).props(
                        f"no-caps size=md color={color}"
                        + (" outline" if not is_selected else "")
                    ).classes("rounded-lg")
                    if is_selected:
                        btn.style("font-weight: 700;")

                    # X button to remove non-running sessions
                    if not is_running:
                        ui.button(
                            icon="close",
                            on_click=lambda _, _sid=sid: on_remove(_sid),
                        ).props(
                            "round dense size=sm color=red-4"
                        ).classes("-ml-2").style(
                            "min-width: 24px; min-height: 24px; padding: 2px;"
                        )

    return container, update
