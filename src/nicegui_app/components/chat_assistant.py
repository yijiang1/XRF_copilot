"""LLM-powered chat assistant for XRF parameter configuration."""

import re
import httpx
from nicegui import ui
from ..state import AppState
from ..config import ANL_USERNAME, ARGO_BASE_URL
from ..services.llm_service import ChatAssistantService


def _is_argo_reachable() -> bool:
    """Quick check whether the Argo Gateway endpoint is reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(ARGO_BASE_URL)
            return resp.status_code < 500
    except Exception:
        return False


def create_chat_assistant(state: AppState, elements_ref: dict) -> None:
    """Create the LLM chat assistant UI.

    Placed inside the Configuration Parameters card, before the
    expansion sections. Uses elements_ref (a mutable dict) so that
    parameter application works even though input_elements are
    populated after this function runs.

    Args:
        state: Shared application state.
        elements_ref: {"elements": dict, "valid": list} populated
                      after create_parameter_form() returns.
    """
    service = ChatAssistantService(automation_level="free")
    pending_params = {"value": None}

    # --- Title ---
    ui.label(
        "Fluoro (XRF Simulation Assistant)"
    ).classes("text-sm font-bold py-1")

    # --- Mode radio + Reset ---
    with ui.row().classes("w-full items-center justify-between"):
        level_radio = ui.radio(
            {
                "free": "Free-form mode",
                "inquiry": "Inquiry mode",
                "suggest": "Suggestion mode",
            },
            value="free",
        ).props("inline dense")

        reset_btn = ui.button("Reset", icon="refresh").props(
            "flat dense size=sm"
        )

    # --- Chat scroll area ---
    with ui.scroll_area().classes(
        "w-full h-64 border rounded bg-white"
    ) as scroll:
        chat_container = ui.column().classes("w-full gap-1 p-2")

    def _scroll_to_bottom():
        ui.run_javascript(f"""
            const el = document.getElementById('c{scroll.id}');
            if (el) {{
                const c = el.querySelector('.q-scrollarea__container');
                if (c) c.scrollTop = c.scrollHeight;
            }}
        """)

    def _add_bot_message(text: str):
        with chat_container:
            ui.chat_message(text=text, name="Fluoro", sent=False)
        _scroll_to_bottom()

    def _add_user_message(text: str):
        with chat_container:
            ui.chat_message(text=text, name="You", sent=True)
        _scroll_to_bottom()

    # --- Input row ---
    with ui.row().classes("w-full items-center gap-2 mt-1"):
        text_input = ui.input(
            placeholder="Describe your XRF experiment..."
        ).classes("flex-grow").props("dense outlined")
        send_btn = ui.button("Send", icon="send").props("dense color=primary")
        apply_btn = ui.button("Apply", icon="check").props(
            "dense color=positive"
        )
        apply_btn.set_visibility(False)

    # --- Apply suggested parameters to the form ---
    def _apply_params():
        suggested = pending_params["value"]
        if not suggested:
            return

        input_elements = elements_ref["elements"]
        valid_params = elements_ref["valid"]

        applied = []
        skipped = []
        for param_name, param_value in suggested.items():
            if param_name in input_elements and param_name in valid_params:
                el = input_elements[param_name]
                # Type coercion
                if isinstance(el, ui.number):
                    if isinstance(param_value, str):
                        param_value = (
                            float(param_value)
                            if "." in param_value
                            else int(param_value)
                        )
                elif isinstance(el, ui.switch):
                    param_value = bool(param_value)
                elif isinstance(el, ui.input):
                    # For elements field, convert list to comma-separated string
                    if isinstance(param_value, list):
                        param_value = ", ".join(str(v) for v in param_value)
                    else:
                        param_value = str(param_value)
                el.set_value(param_value)
                applied.append(f"{param_name} = {param_value!r}")
            else:
                skipped.append(param_name)

        parts = []
        if applied:
            parts.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
        if skipped:
            parts.append("Skipped (unknown): " + ", ".join(skipped))
        _add_bot_message("\n".join(parts) if parts else "No parameters to apply.")

        pending_params["value"] = None
        apply_btn.set_visibility(False)

    apply_btn.on_click(_apply_params)

    # --- Send message ---
    async def _on_send():
        user_text = text_input.value.strip()
        if not user_text:
            return

        text_input.set_value("")
        _add_user_message(user_text)
        text_input.disable()
        send_btn.disable()

        try:
            response_text, suggested = await service.send_message(user_text)

            # Strip params block from display
            display_text = response_text
            if suggested:
                display_text = re.sub(
                    r"```params\s*\n.*?\n```", "", display_text, flags=re.DOTALL
                ).strip()
                if not display_text:
                    display_text = re.sub(
                        r"```json\s*\n.*?\n```", "", response_text, flags=re.DOTALL
                    ).strip()
                if not display_text:
                    display_text = re.sub(
                        r"```\s*\n.*?\n```", "", response_text, flags=re.DOTALL
                    ).strip()

            _add_bot_message(display_text or "Parameters extracted.")

            if suggested:
                summary = "\n".join(
                    f"  {k}: {v!r}" for k, v in suggested.items()
                )
                _add_bot_message(f"Suggested parameters:\n{summary}")
                pending_params["value"] = suggested
                apply_btn.set_visibility(True)

        except Exception as e:
            _add_bot_message(f"Error: {e}")

        finally:
            text_input.enable()
            send_btn.enable()

    send_btn.on_click(_on_send)
    text_input.on("keydown.enter", _on_send)

    # --- Level change ---
    def _on_level_change(e):
        service.set_automation_level(e.value)
        service.reset()
        chat_container.clear()
        pending_params["value"] = None
        apply_btn.set_visibility(False)
        _add_bot_message(service.get_greeting())

    level_radio.on_value_change(_on_level_change)

    # --- Reset ---
    def _on_reset():
        service.reset()
        chat_container.clear()
        pending_params["value"] = None
        apply_btn.set_visibility(False)
        _add_bot_message(service.get_greeting())

    reset_btn.on_click(_on_reset)

    # --- Initial state ---
    if not ANL_USERNAME:
        _add_bot_message(
            "Chat assistant disabled — ANL_USERNAME not configured."
        )
        text_input.disable()
        send_btn.disable()
    elif not _is_argo_reachable():
        _add_bot_message(
            "Chat assistant disabled — Argo Gateway is not reachable. "
            "This feature requires Argonne network access."
        )
        text_input.disable()
        send_btn.disable()
    else:
        _add_bot_message(service.get_greeting())
