"""Real-time reconstruction result viewer.

Displays 2D slices from the latest checkpoint HDF5 during or after
reconstruction.  Fetches data from the backend via HTTP (works across
machines), following the ptycho-gui-mic pattern.

UI: section header + element dropdown + slice slider + image display.
"""

import asyncio
from nicegui import ui


# ── Zoom / pan JS (same as h5_inspector — Ctrl+wheel zoom, drag pan) ─────────
_ZOOM_JS = """<script>
window._initZoomable = window._initZoomable || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || container._zoomInit) return;
    container._zoomInit = true;
    var state = {scale:1, panX:0, panY:0, dragging:false, lastX:0, lastY:0};
    container._zoomState = state;
    function apply() {
        var img = container.querySelector('img');
        if (!img) return;
        img.style.transformOrigin = '0 0';
        img.style.transform = 'matrix('+state.scale+',0,0,'+state.scale+','+state.panX+','+state.panY+')';
        container.style.cursor = state.scale > 1 ? 'grab' : '';
    }
    container.addEventListener('wheel', function(e) {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        var rect = container.getBoundingClientRect();
        var mx = e.clientX - rect.left, my = e.clientY - rect.top;
        var factor = e.deltaY < 0 ? 1.15 : 1/1.15;
        var newScale = Math.max(0.5, Math.min(10, state.scale * factor));
        state.panX = mx - (mx - state.panX) * (newScale / state.scale);
        state.panY = my - (my - state.panY) * (newScale / state.scale);
        state.scale = newScale;
        if (state.scale > 0.95 && state.scale < 1.05) { state.scale=1; state.panX=0; state.panY=0; }
        apply();
    }, {passive: false});
    container.addEventListener('mousedown', function(e) {
        if (state.scale <= 1) return;
        state.dragging = true; state.lastX = e.clientX; state.lastY = e.clientY;
        container.style.cursor = 'grabbing'; e.preventDefault();
    });
    container.addEventListener('mousemove', function(e) {
        if (!state.dragging) return;
        state.panX += e.clientX - state.lastX; state.panY += e.clientY - state.lastY;
        state.lastX = e.clientX; state.lastY = e.clientY; apply();
    });
    function endDrag() { state.dragging = false; container.style.cursor = state.scale > 1 ? 'grab' : ''; }
    container.addEventListener('mouseup', endDrag);
    container.addEventListener('mouseleave', endDrag);
    container.addEventListener('dblclick', function() { state.scale=1; state.panX=0; state.panY=0; apply(); });
};
</script>"""


def create_result_viewer() -> tuple[ui.element, callable]:
    """Create a reconstruction result slice viewer.

    Returns:
        (container, async update_fn(api, session_id)) tuple.
        Call update_fn each poll cycle to check for new checkpoints.
    """
    collapsed = {"value": True}   # start collapsed (no data yet)
    live = {"enabled": True}
    state = {
        "elements": [],
        "n_slices": 0,
        "iteration": -1,
        "mtime": 0,            # file mtime — detects in-place changes to recon.h5
        "session_id": "",
        "last_file": "",       # avoid redundant fetches
        "elem_idx": 0,
        "slice_idx": 0,
    }
    _seq = {"n": 0}  # sequence counter for debounce
    _play = {"active": False, "btn": None}  # auto-play state

    ui.add_head_html(_ZOOM_JS)

    # ── Section header ────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center justify-between mb-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("visibility", size="xs").classes("text-gray-400")
            ui.label("Results").classes("section-header").style("margin-bottom: 0;")
            iter_label = ui.label("").classes("text-xs text-gray-400 italic")

        with ui.row().classes("items-center gap-1"):
            def _toggle_live():
                live["enabled"] = not live["enabled"]
                if live["enabled"]:
                    live_btn.props("color=green flat dense no-caps size=sm")
                    live_btn.set_text("Live")
                else:
                    live_btn.props("color=grey flat dense no-caps size=sm")
                    live_btn.set_text("Paused")

            live_btn = ui.button("Live", icon="circle", on_click=_toggle_live).props(
                "color=green flat dense no-caps size=sm"
            )

            def _toggle_collapse():
                collapsed["value"] = not collapsed["value"]
                content_area.set_visibility(not collapsed["value"])
                chevron_btn.props(
                    "icon=expand_more flat dense size=sm color=grey"
                    if collapsed["value"] else
                    "icon=expand_less flat dense size=sm color=grey"
                )

            chevron_btn = ui.button(icon="expand_more", on_click=_toggle_collapse).props(
                "flat dense size=sm color=grey"
            )

    # ── Content area (collapsible) ────────────────────────────────────────────
    content_area = ui.column().classes("w-full gap-2")
    content_area.set_visibility(False)  # hidden until first checkpoint

    with content_area:
        # File path label (below header, above controls)
        file_path_label = ui.label("").classes("text-xs text-gray-400 font-mono truncate w-full")

        # Controls row: element dropdown + slice slider
        with ui.row().classes("w-full items-end gap-4"):
            elem_select = ui.select(
                label="Element",
                options=[],
                value=None,
            ).classes("w-40")

            with ui.column().classes("flex-1 gap-0"):
                with ui.row().classes("w-full items-center gap-1"):
                    slice_label = ui.label("Slice: 0 / 0").classes("text-sm text-gray-600")
                    ui.space()
                    stats_label = ui.label("").classes("text-xs text-gray-400")

                slider_container = ui.row().classes("w-full items-center gap-1")

        # Image display — plain <img> injected via JS (avoids q-img aspect-ratio
        # complications, same pattern as h5_inspector).
        _cid = f"result-viewer-{id(content_area)}"
        with ui.element("div").style(
            "position: relative; overflow: hidden; background: #1e1e1e; "
            "border-radius: 6px; max-height: 500px; width: 100%; max-width: 600px;"
        ) as img_container:
            img_container._props["id"] = _cid
        # No ui.image() — we inject a plain <img> via JS below
        img_el = None  # placeholder; image src set via JS

        async def _init_img_js():
            await ui.run_javascript(
                f"(function(){{"
                f"var c=document.getElementById('{_cid}');"
                f"if(!c||c.querySelector('img'))return;"
                f"var img=document.createElement('img');"
                f"img.style.cssText='width:100%;height:auto;display:block;"
                f"image-rendering:pixelated;';"
                f"c.appendChild(img);"
                f"}})()"
            )
            await ui.run_javascript(f"window._initZoomable('{_cid}')")

        ui.timer(0.5, _init_img_js, once=True)

        no_data_label = ui.label("No reconstruction data available yet").classes(
            "text-sm text-gray-400 italic"
        )

    # ── Internal: fetch and display a slice ────────────────────────────────────

    async def _fetch_slice(api, session_id):
        """Fetch the current element+slice from the backend and update image."""
        try:
            resp = await api.get_recon_slice(
                session_id,
                elem_idx=state["elem_idx"],
                slice_idx=state["slice_idx"],
                file=state["last_file"],
            )
            if resp.get("status") == "ok" and resp.get("data_url"):
                no_data_label.set_visibility(False)
                img_container.set_visibility(True)
                vmin = resp.get("vmin", 0)
                vmax = resp.get("vmax", 0)
                vmean = resp.get("vmean", 0)
                ny = resp.get("ny", 0)
                nx = resp.get("nx", 0)
                stats_label.set_text(
                    f"min={vmin:.4g}  max={vmax:.4g}  mean={vmean:.4g}  ({ny}x{nx})"
                )
                # Update plain <img> src via JS (same pattern as h5_inspector)
                data_url = resp["data_url"]
                ui.run_javascript(
                    f"(function(){{"
                    f"var c=document.getElementById('{_cid}');"
                    f"if(!c)return;"
                    f"var img=c.querySelector('img');"
                    f"if(!img)return;"
                    f"var tmp=new Image();"
                    f"tmp.onload=function(){{img.src=tmp.src;}};"
                    f"tmp.src='{data_url}';"
                    f"c.style.maxWidth=Math.max(256,Math.min(600,{nx}*5))+'px';"
                    f"}})()"
                )
        except Exception:
            pass

    async def _debounced_fetch(api, session_id):
        """Debounced slice fetch (200ms delay for slider drag).

        Uses sequence-counter pattern: if a newer event arrives during the
        sleep, the stale coroutine simply returns without fetching.
        """
        seq = _seq["n"] = _seq["n"] + 1
        await asyncio.sleep(0.20)
        if _seq["n"] == seq:
            await _fetch_slice(api, session_id)

    def _stop_playback():
        """Stop auto-play if active and reset button icon."""
        if _play["active"]:
            _play["active"] = False
            if _play["btn"]:
                _play["btn"].props("icon=play_arrow flat dense round size=sm")

    # Store api ref for callbacks
    _api_ref = {"api": None, "sid": ""}

    # ── Main update function (called each poll cycle) ──────────────────────────

    async def update(api, session_id):
        """Check for new checkpoint and update viewer if needed."""
        _api_ref["api"] = api
        _api_ref["sid"] = session_id

        if not live["enabled"]:
            return
        if not session_id:
            return

        try:
            info = await api.get_session_recon_info(session_id)
        except Exception:
            return

        if info.get("status") != "ok":
            return

        elements = info.get("elements", [])
        n_slices = info.get("n_slices", 0)
        iteration = info.get("iteration", -1)
        latest_file = info.get("file", "")
        mtime = info.get("mtime", 0)

        # Auto-expand on first data
        if collapsed["value"] and elements:
            collapsed["value"] = False
            content_area.set_visibility(True)
            chevron_btn.props("icon=expand_less flat dense size=sm color=grey")

        no_data_label.set_visibility(False)

        # Update iteration label and file path
        iter_label.set_text(f"(iteration {iteration})" if iteration >= 0 else "")
        if latest_file:
            file_path_label.set_text(f"File: {latest_file}")

        # Update element dropdown if elements changed
        if elements != state["elements"]:
            state["elements"] = elements
            options = {i: elem for i, elem in enumerate(elements)}
            elem_select.options = options
            elem_select.update()
            if state["elem_idx"] >= len(elements):
                state["elem_idx"] = 0
            elem_select.set_value(state["elem_idx"])

        # Update slice slider if n_slices changed
        if n_slices != state["n_slices"] and n_slices > 0:
            state["n_slices"] = n_slices
            if state["slice_idx"] >= n_slices:
                state["slice_idx"] = n_slices // 2
            slice_label.set_text(f"Slice: {state['slice_idx']} / {n_slices - 1}")

            # Rebuild slider
            _stop_playback()
            slider_container.clear()
            with slider_container:
                _from_button = {"skip": False}

                async def _on_prev():
                    _stop_playback()
                    if state["slice_idx"] > 0:
                        state["slice_idx"] -= 1
                        _from_button["skip"] = True
                        sl_ref[0].set_value(state["slice_idx"])
                        slice_label.set_text(
                            f"Slice: {state['slice_idx']} / {state['n_slices'] - 1}"
                        )
                        if _api_ref["api"]:
                            await _fetch_slice(_api_ref["api"], _api_ref["sid"])

                async def _on_next():
                    _stop_playback()
                    if state["slice_idx"] < state["n_slices"] - 1:
                        state["slice_idx"] += 1
                        _from_button["skip"] = True
                        sl_ref[0].set_value(state["slice_idx"])
                        slice_label.set_text(
                            f"Slice: {state['slice_idx']} / {state['n_slices'] - 1}"
                        )
                        if _api_ref["api"]:
                            await _fetch_slice(_api_ref["api"], _api_ref["sid"])

                async def _toggle_play():
                    if _play["active"]:
                        _stop_playback()
                        return
                    _play["active"] = True
                    _play["btn"].props("icon=pause flat dense round size=sm")
                    while _play["active"] and _api_ref["api"]:
                        # Advance to next slice (loop back at end)
                        if state["slice_idx"] >= state["n_slices"] - 1:
                            state["slice_idx"] = 0
                        else:
                            state["slice_idx"] += 1
                        _from_button["skip"] = True
                        sl_ref[0].set_value(state["slice_idx"])
                        slice_label.set_text(
                            f"Slice: {state['slice_idx']} / {state['n_slices'] - 1}"
                        )
                        await _fetch_slice(_api_ref["api"], _api_ref["sid"])
                        await asyncio.sleep(0.30)
                    _stop_playback()

                play_btn = ui.button(icon="play_arrow", on_click=_toggle_play).props(
                    "flat dense round size=sm"
                )
                _play["btn"] = play_btn

                ui.button(icon="chevron_left", on_click=_on_prev).props(
                    "flat dense round size=sm"
                )

                async def _on_slider_change(e):
                    state["slice_idx"] = int(e.value)
                    slice_label.set_text(
                        f"Slice: {state['slice_idx']} / {state['n_slices'] - 1}"
                    )
                    if _from_button["skip"]:
                        _from_button["skip"] = False
                        return
                    _stop_playback()
                    if _api_ref["api"]:
                        await _debounced_fetch(_api_ref["api"], _api_ref["sid"])

                sl = ui.slider(
                    min=0,
                    max=n_slices - 1,
                    step=1,
                    value=state["slice_idx"],
                    on_change=_on_slider_change,
                ).classes("flex-1")
                sl_ref = [sl]

                ui.button(icon="chevron_right", on_click=_on_next).props(
                    "flat dense round size=sm"
                )

        # Fetch slice if checkpoint changed, file updated in-place, or session changed
        need_fetch = (
            latest_file != state["last_file"]
            or session_id != state["session_id"]
            or iteration != state["iteration"]
            or mtime != state["mtime"]  # detects in-place recon.h5 updates
        )
        if need_fetch:
            state["last_file"] = latest_file
            state["session_id"] = session_id
            state["iteration"] = iteration
            state["mtime"] = mtime
            await _fetch_slice(api, session_id)

    # Element dropdown change handler
    async def _on_elem_change(e):
        if e.value is not None:
            state["elem_idx"] = int(e.value)
            if _api_ref["api"]:
                await _fetch_slice(_api_ref["api"], _api_ref["sid"])

    elem_select.on_value_change(_on_elem_change)

    return content_area, update
