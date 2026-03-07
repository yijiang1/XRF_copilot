"""NiceGUI frontend entry point for XRF Simulation Copilot."""

import argparse
import re
import httpx
from nicegui import ui
from .config import HOST, PORT, BACKEND_API_KEY

from .pages.simulation import create_simulation_page
from .pages.reconstruction_all import create_reconstruction_all_page
from .pages.method_explanation import create_method_explanation_page

_APS_SR_STATUS_URL = "https://www3.aps.anl.gov/aod/blops/status/srStatus.html"

# Runtime API key (set via CLI, overrides .env)
_runtime_api_key: str = ""

# ── Favicon: XRF "X" on deep-red background ──
_FAVICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%23b91c1c'/>"
    "<path d='M14,14 L22,14 L32,26 L42,14 L50,14 L38,32 L50,50 L42,50"
    " L32,38 L22,50 L14,50 L26,32 Z' fill='white'/>"
    "</svg>"
)

# ── Global CSS applied to every page ──
_GLOBAL_CSS = """
/* ── Consistent card styling ── */
.q-card {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03) !important;
}
.q-dialog .q-card {
    box-shadow: 0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08) !important;
    border-radius: 14px !important;
    background: white !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Page content wrapper ── */
.page-content {
    padding: 16px 24px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
}

/* ── Section header inside cards ── */
.section-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 12px;
}

/* ── Consistent input field rounding ── */
.q-field--outlined .q-field__control {
    border-radius: 8px;
}

/* ── Nav header ── */
.nav-header {
    background: #1e293b;
    border-bottom: 1px solid #334155;
    min-height: 48px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.nav-header .q-btn {
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    font-size: 1.05rem !important;
}
.nav-header .q-btn.nav-btn-active {
    background: rgba(255, 255, 255, 0.10) !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.6) !important;
    border-radius: 6px 6px 0 0 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.nav-header .q-btn.nav-btn-inactive {
    color: rgba(255, 255, 255, 0.55) !important;
}
.nav-header .q-btn.nav-btn-inactive:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #ffffff !important;
}

/* ── Body background ── */
.app-body {
    background: #f8fafc !important;
}
"""


def _get_api_key() -> str:
    """Return the active backend API key (runtime flag > .env > empty)."""
    return _runtime_api_key or BACKEND_API_KEY


def _unauthorized():
    """Render a minimal unauthorized page."""
    with ui.column().classes("items-center justify-center w-full h-screen gap-4"):
        ui.icon("lock", size="4rem").classes("text-red-400")
        ui.label("Unauthorized").classes("text-3xl font-bold text-red-500")
        ui.label("Navigate to /<api-key> to access the app.").classes("text-gray-500")


def _create_nav_header(active_page: str, key: str):
    """Render the dark top navigation bar.

    *active_page* is the page identifier string (e.g. ``"simulation"``).
    *key* is the API key embedded in the URL for navigation links.
    """
    ui.add_css(_GLOBAL_CSS)
    ui.colors(
        primary="#dc2626",
        secondary="#f97316",
        accent="#7c3aed",
        positive="#10b981",
        negative="#ef4444",
        warning="#f59e0b",
    )

    with ui.header().classes("items-center gap-2 px-6 nav-header"):
        # Brand name / logo
        with ui.row().classes("items-center gap-2 q-mr-lg"):
            ui.icon("biotech", size="1.6rem").classes("text-red-400")
            ui.label("XRF Copilot").classes("text-white font-bold text-lg")

        nav_items = [
            ("Reconstruction", "layers", "reconstruction_all", f"/{key}"),
            ("Simulation", "science", "simulation", f"/{key}/simulation"),
            ("Method", "menu_book", "method_explanation", f"/{key}/method-bnl"),
        ]
        for label, icon, page_id, path in nav_items:
            btn = ui.button(
                label,
                icon=icon,
                on_click=lambda p=path: ui.navigate.to(p),
            ).props("flat no-caps size=md color=white")
            btn.classes("nav-btn-active" if page_id == active_page else "nav-btn-inactive")

        ui.space()

        # ── APS beam status badge ──
        beam_badge = ui.label("APS: —").classes(
            "text-sm font-semibold px-3 py-1 rounded-full self-center "
            "bg-slate-700 text-slate-300"
        )

        async def _poll_beam_status():
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(_APS_SR_STATUS_URL)
                    html = resp.text
                m = re.search(r"Operations Status.*?<b>\s*([^<]+?)\s*</b>", html, re.DOTALL)
                if m:
                    op_status = m.group(1).strip()
                    has_beam = bool(re.search(r"Delivered\s+Beam", op_status, re.IGNORECASE))
                    beam_badge.set_text(f"APS: {op_status}")
                    beam_badge.classes(
                        remove="bg-slate-700 text-slate-300 bg-red-900 text-red-300 bg-green-900 text-green-300",
                        add="bg-green-900 text-green-300" if has_beam else "bg-red-900 text-red-300",
                    )
                else:
                    beam_badge.set_text("APS: —")
                    beam_badge.classes(
                        remove="bg-green-900 text-green-300 bg-red-900 text-red-300",
                        add="bg-slate-700 text-slate-300",
                    )
            except Exception:
                beam_badge.set_text("APS: —")
                beam_badge.classes(
                    remove="bg-green-900 text-green-300 bg-red-900 text-red-300",
                    add="bg-slate-700 text-slate-300",
                )

        ui.timer(60.0, _poll_beam_status, immediate=True)


# ── Routes ──

@ui.page("/", title="XRF Simulation Copilot", favicon=_FAVICON)
def index_bare():
    """Root with no key → show unauthorized."""
    _unauthorized()


@ui.page("/{key}", title="XRF Reconstruction", favicon=_FAVICON)
def index(key: str):
    """Landing page — reconstruction (BNL, Panpan, Wendy)."""
    if key != _get_api_key():
        _unauthorized()
        return
    ui.query("body").classes("app-body")
    _create_nav_header("reconstruction_all", key)
    create_reconstruction_all_page(api_key=_get_api_key())


@ui.page("/{key}/simulation", title="XRF Simulation Copilot", favicon=_FAVICON)
def simulation(key: str):
    """Simulation page."""
    if key != _get_api_key():
        _unauthorized()
        return
    ui.query("body").classes("app-body")
    _create_nav_header("simulation", key)
    create_simulation_page(api_key=_get_api_key())


@ui.page("/{key}/method-bnl", title="Method Explanation (BNL)", favicon=_FAVICON)
def method_explanation(key: str):
    """Step-by-step explanation of the BNL self-absorption correction algorithm."""
    if key != _get_api_key():
        _unauthorized()
        return
    ui.query("body").classes("app-body")
    _create_nav_header("method_explanation", key)
    create_method_explanation_page()


def run():
    global _runtime_api_key

    parser = argparse.ArgumentParser(description="XRF Simulation Copilot GUI")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Backend API key — also used as the URL path key",
    )
    args, _ = parser.parse_known_args()

    if args.api_key:
        _runtime_api_key = args.api_key
        print(f"Backend API key set via --api-key flag.")
        print(f"  → Open: http://{HOST}:{PORT}/{args.api_key}  (Reconstruction)")
        print(f"          http://{HOST}:{PORT}/{args.api_key}/simulation")

    ui.run(
        host=HOST,
        port=PORT,
        reload=False,
        show=False,
        title="XRF Simulation Copilot",
        storage_secret="xrf-copilot-storage",
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
