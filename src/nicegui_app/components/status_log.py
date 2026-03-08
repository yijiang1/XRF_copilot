"""Scrollable monospace status log component — incremental update pattern.

Uses a plain <div> with overflow-y:auto instead of Quasar's q-scroll-area
so that native text selection and copy work reliably.
"""

from nicegui import ui
from ..state import AppState

MAX_DISPLAY = 500    # max messages kept in DOM
TRIM_THRESHOLD = 700  # trim to MAX_DISPLAY when exceeded


def create_status_log(state: AppState, on_clear=None) -> tuple:
    """Create a scrollable monospace log display with incremental updates.

    Args:
        state:    AppState whose .messages list is rendered.
        on_clear: Optional callback invoked after the log is cleared (use to
                  reset per-method message lists and log-offset counters).

    Returns:
        (log_element, update_log_fn) tuple.
    """
    collapsed = {"value": False}
    live_state = {"enabled": True}

    # ── Header row ─────────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center justify-between mb-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("terminal", size="xs").classes("text-gray-400")
            ui.label("Status Log").classes("section-header").style("margin-bottom: 0;")
            ui.label("(refreshes every 3s)").classes("text-xs text-gray-400 italic")

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

            def _copy_log():
                # navigator.clipboard requires HTTPS; use textarea fallback
                # for HTTP origins (typical for internal LAN apps).
                ui.run_javascript(f"""
                    const el = document.getElementById('c{log_container.id}');
                    if (el) {{
                        const text = el.innerText;
                        if (navigator.clipboard && window.isSecureContext) {{
                            navigator.clipboard.writeText(text);
                        }} else {{
                            const ta = document.createElement('textarea');
                            ta.value = text;
                            ta.style.position = 'fixed';
                            ta.style.left = '-9999px';
                            document.body.appendChild(ta);
                            ta.select();
                            document.execCommand('copy');
                            document.body.removeChild(ta);
                        }}
                    }}
                """)
                ui.notify("Log copied to clipboard", type="positive", position="top", timeout=2000)

            ui.button("Copy", icon="content_copy", on_click=_copy_log).props(
                "flat dense no-caps size=sm color=grey"
            )

            def _save_log():
                ui.run_javascript(f"""
                    const el = document.getElementById('c{log_container.id}');
                    if (el) {{
                        const text = el.innerText;
                        if (!text || !text.trim()) return;
                        const blob = new Blob([text], {{ type: 'text/plain' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '_');
                        a.download = 'xrf_log_' + ts + '.txt';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    }}
                """)

            ui.button("Save", icon="save_alt", on_click=_save_log).props(
                "flat dense no-caps size=sm color=grey"
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
                log_div.set_visibility(not collapsed["value"])
                chevron_btn.props(
                    "icon=expand_more flat dense size=sm color=grey"
                    if collapsed["value"] else
                    "icon=expand_less flat dense size=sm color=grey"
                )

            chevron_btn = ui.button(icon="expand_less", on_click=_toggle_collapse).props(
                "flat dense size=sm color=grey"
            )

    # ── Log container — plain div for native text selection ───────────────────
    log_div = ui.element("div").classes("w-full rounded").style(
        "background: #f8fafc; border: 1px solid #e5e7eb; "
        "overflow-y: auto; height: 16rem; "
        "user-select: text; cursor: text;"
    )
    with log_div:
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

        # Auto-scroll to bottom (native scrollTop on plain div)
        ui.run_javascript(f"""
            const el = document.getElementById('c{log_div.id}');
            if (el) el.scrollTop = el.scrollHeight;
        """)

    return log_div, update_log
