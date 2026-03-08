"""Scrollable monospace status log component — incremental update pattern."""

from nicegui import ui
from ..state import AppState

MAX_DISPLAY = 500    # max messages kept in DOM
TRIM_THRESHOLD = 700  # trim to MAX_DISPLAY when exceeded


def create_status_log(state: AppState, on_clear=None) -> tuple[ui.scroll_area, callable]:
    """Create a scrollable monospace log display with incremental updates.

    Args:
        state:    AppState whose .messages list is rendered.
        on_clear: Optional callback invoked after the log is cleared (use to
                  reset per-method message lists and log-offset counters).

    Returns:
        (scroll_area, update_log_fn) tuple.
    """
    collapsed = {"value": False}
    live_state = {"enabled": True}

    # ── Header row ─────────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center justify-between mb-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("terminal", size="xs").classes("text-gray-400")
            ui.label("Status Log").classes("section-header").style("margin-bottom: 0;")
            ui.label("(refreshes every 2s)").classes("text-xs text-gray-400 italic")

        with ui.row().classes("items-center gap-1"):

            def _toggle_live():
                live_state["enabled"] = not live_state["enabled"]
                if live_state["enabled"]:
                    live_btn.props("color=green flat dense no-caps size=sm")
                    live_btn.set_text("Live")
                    update_log()
                else:
                    live_btn.props("color=grey flat dense no-caps size=sm")
                    live_btn.set_text("Paused")

            live_btn = ui.button("Live", icon="circle", on_click=_toggle_live).props(
                "color=green flat dense no-caps size=sm"
            )

            def _clear_log():
                state.messages.clear()
                log_container.clear()
                dom_count["value"] = 0
                if on_clear:
                    on_clear()

            ui.button("Clear", icon="delete_outline", on_click=_clear_log).props(
                "flat dense no-caps size=sm color=grey"
            )

            def _toggle_collapse():
                collapsed["value"] = not collapsed["value"]
                scroll.set_visibility(not collapsed["value"])
                chevron_btn.props(
                    "icon=expand_more flat dense size=sm color=grey"
                    if collapsed["value"] else
                    "icon=expand_less flat dense size=sm color=grey"
                )

            chevron_btn = ui.button(icon="expand_less", on_click=_toggle_collapse).props(
                "flat dense size=sm color=grey"
            )

    # ── Scroll area ────────────────────────────────────────────────────────────
    with ui.scroll_area().classes("w-full h-64 rounded").style(
        "background: #f8fafc; border: 1px solid #e5e7eb;"
    ) as scroll:
        log_container = ui.column().classes("w-full gap-0 p-2")

    dom_count = {"value": 0}  # messages currently rendered in DOM

    def update_log():
        if not live_state["enabled"]:
            return  # paused — skip DOM update

        n = len(state.messages)

        # Memory management: trim in-place when list grows too large
        if n > TRIM_THRESHOLD:
            state.messages[:] = state.messages[-MAX_DISPLAY:]
            n = len(state.messages)

        if n == dom_count["value"]:
            return  # nothing new

        if n < dom_count["value"]:
            # Slow path: list shrank (CLEAR or trim) — full rebuild
            log_container.clear()
            with log_container:
                for msg in state.messages:
                    ui.html(msg, sanitize=False).classes("font-mono text-sm whitespace-pre-wrap")
        else:
            # Fast path: append only the new messages
            with log_container:
                for msg in state.messages[dom_count["value"]:]:
                    ui.html(msg, sanitize=False).classes("font-mono text-sm whitespace-pre-wrap")

        dom_count["value"] = n

        # Auto-scroll to bottom
        ui.run_javascript(f"""
            const scrollArea = document.getElementById('c{scroll.id}');
            if (scrollArea) {{
                const container = scrollArea.querySelector('.q-scrollarea__container');
                if (container) container.scrollTop = container.scrollHeight;
            }}
        """)

    return scroll, update_log
