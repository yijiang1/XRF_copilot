"""HDF5 data inspector: slice-based viewer for large XRF projection files.

Reads only metadata (elements, thetas) on file load, then fetches individual
2D slices (~1.4 MB each) on demand — never loads the full array into memory.

Images are rendered as Viridis PNG (2–98 percentile normalisation) and
displayed via a plain <img> element with interactive zoom/pan/ruler (no Plotly).
"""

import os
import io
import asyncio
import base64
import h5py
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui, run


# ── Zoom / pan JavaScript (CSS matrix transform, Ctrl+wheel) ─────────────────
_ZOOM_JS = """<script>
window._initZoomable = window._initZoomable || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || container._zoomInit) return;
    container._zoomInit = true;

    var state = {scale:1, panX:0, panY:0, dragging:false, lastX:0, lastY:0};
    container._zoomState = state;

    function applyTo(c, s) {
        var img = c.querySelector('img');
        if (!img) return;
        img.style.transformOrigin = '0 0';
        img.style.transform = 'matrix('+s.scale+',0,0,'+s.scale+','+s.panX+','+s.panY+')';
        c.style.cursor = s.scale > 1 ? 'grab' : '';
    }
    function apply() {
        applyTo(container, state);
        if (window._drawRuler) window._drawRuler(containerId);
        if (window._drawCropRect) window._drawCropRect(containerId);
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
        if (container._rulerActive || container._cropActive || state.scale <= 1) return;
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

window._zoomTo = window._zoomTo || function(containerId, factor) {
    var container = document.getElementById(containerId);
    if (!container || !container._zoomState) return;
    var state = container._zoomState;
    var rect = container.getBoundingClientRect();
    var cx = rect.width/2, cy = rect.height/2;
    var newScale = Math.max(0.5, Math.min(10, state.scale * factor));
    state.panX = cx - (cx - state.panX) * (newScale / state.scale);
    state.panY = cy - (cy - state.panY) * (newScale / state.scale);
    state.scale = newScale;
    if (state.scale > 0.95 && state.scale < 1.05) { state.scale=1; state.panX=0; state.panY=0; }
    var img = container.querySelector('img');
    if (img) { img.style.transformOrigin='0 0'; img.style.transform='matrix('+state.scale+',0,0,'+state.scale+','+state.panX+','+state.panY+')'; }
    container.style.cursor = state.scale > 1 ? 'grab' : '';
    if (window._drawRuler) window._drawRuler(containerId);
    if (window._drawCropRect) window._drawCropRect(containerId);
};

window._zoomReset = window._zoomReset || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !container._zoomState) return;
    var state = container._zoomState;
    state.scale=1; state.panX=0; state.panY=0;
    var img = container.querySelector('img');
    if (img) img.style.transform = '';
    container.style.cursor = '';
    if (window._drawRuler) window._drawRuler(containerId);
    if (window._drawCropRect) window._drawCropRect(containerId);
};
</script>"""

# ── Ruler / measurement JavaScript (SVG overlay) ─────────────────────────────
_RULER_JS = """<script>
window._initRuler = window._initRuler || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || container._rulerInit) return;
    container._rulerInit = true;
    container.style.position = 'relative';
    var svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;';
    container.appendChild(svg);
    container._rulerSvg = svg;
    container._rulerMode = false;
    container._rulerState = null;
};

window._setRulerMode = window._setRulerMode || function(containerId, enabled) {
    var container = document.getElementById(containerId);
    if (!container) return;
    if (!container._rulerInit) window._initRuler(containerId);
    container._rulerMode = enabled;
    var svg = container._rulerSvg;
    svg.style.pointerEvents = enabled ? 'all' : 'none';
    if (enabled) {
        container._rulerActive = true;
        container._rulerMouseDown = function(e) {
            e.preventDefault(); e.stopPropagation();
            var rect = container.getBoundingClientRect();
            var zs = container._zoomState || {scale:1,panX:0,panY:0};
            var sx = e.clientX - rect.left, sy = e.clientY - rect.top;
            container._rulerState = {
                imgX1:(sx-zs.panX)/zs.scale, imgY1:(sy-zs.panY)/zs.scale,
                imgX2:(sx-zs.panX)/zs.scale, imgY2:(sy-zs.panY)/zs.scale,
                drawing:true
            };
            window._drawRuler(containerId);
        };
        container._rulerMouseMove = function(e) {
            if (!container._rulerState || !container._rulerState.drawing) return;
            e.preventDefault(); e.stopPropagation();
            var rect = container.getBoundingClientRect();
            var zs = container._zoomState || {scale:1,panX:0,panY:0};
            container._rulerState.imgX2 = (e.clientX-rect.left-zs.panX)/zs.scale;
            container._rulerState.imgY2 = (e.clientY-rect.top -zs.panY)/zs.scale;
            window._drawRuler(containerId);
        };
        container._rulerMouseUp = function(e) {
            if (container._rulerState) container._rulerState.drawing = false;
        };
        svg.addEventListener('mousedown', container._rulerMouseDown);
        svg.addEventListener('mousemove', container._rulerMouseMove);
        svg.addEventListener('mouseup',   container._rulerMouseUp);
    } else {
        container._rulerActive = false;
        if (svg && container._rulerMouseDown) {
            svg.removeEventListener('mousedown', container._rulerMouseDown);
            svg.removeEventListener('mousemove', container._rulerMouseMove);
            svg.removeEventListener('mouseup',   container._rulerMouseUp);
        }
    }
};

window._drawRuler = window._drawRuler || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !container._rulerSvg || !container._rulerState) return;
    var svg = container._rulerSvg;
    var rs = container._rulerState;
    var zs = container._zoomState || {scale:1,panX:0,panY:0};
    svg.innerHTML = '';
    var sx1=rs.imgX1*zs.scale+zs.panX, sy1=rs.imgY1*zs.scale+zs.panY;
    var sx2=rs.imgX2*zs.scale+zs.panX, sy2=rs.imgY2*zs.scale+zs.panY;

    var line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',sx1); line.setAttribute('y1',sy1);
    line.setAttribute('x2',sx2); line.setAttribute('y2',sy2);
    line.setAttribute('stroke','#ff4444'); line.setAttribute('stroke-width','2');
    line.setAttribute('stroke-dasharray','6,4');
    svg.appendChild(line);

    [[sx1,sy1],[sx2,sy2]].forEach(function(pt) {
        var c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx',pt[0]); c.setAttribute('cy',pt[1]);
        c.setAttribute('r','4'); c.setAttribute('fill','#ff4444');
        svg.appendChild(c);
    });

    var meta = container._pixelMeta || {};
    var imgEl = container.querySelector('img');
    var displayWidth = imgEl ? imgEl.offsetWidth : 1;
    var origWidth = meta.image_width || displayWidth;
    var cssToOrig = origWidth / displayWidth;
    var dx = (rs.imgX2-rs.imgX1)*cssToOrig, dy = (rs.imgY2-rs.imgY1)*cssToOrig;
    var origDist = Math.sqrt(dx*dx + dy*dy);
    var pixSizeNm = container._pixelSizeNm || 0;
    var labelText;
    if (pixSizeNm > 0) {
        var physDist = origDist * pixSizeNm;
        labelText = physDist < 1000
            ? physDist.toFixed(1) + ' nm'
            : (physDist / 1000).toFixed(3) + ' \u03bcm';
    } else {
        labelText = Math.round(origDist) + ' px';
    }

    var midX=(sx1+sx2)/2, midY=(sy1+sy2)/2;
    var text = document.createElementNS('http://www.w3.org/2000/svg','text');
    text.textContent = labelText;
    text.setAttribute('x',midX); text.setAttribute('y',midY-8);
    text.setAttribute('text-anchor','middle');
    text.setAttribute('fill','#ff4444'); text.setAttribute('font-size','13');
    text.setAttribute('font-weight','bold'); text.setAttribute('font-family','monospace');
    svg.appendChild(text);

    try {
        var bbox = text.getBBox();
        var bg = document.createElementNS('http://www.w3.org/2000/svg','rect');
        bg.setAttribute('x',bbox.x-3); bg.setAttribute('y',bbox.y-1);
        bg.setAttribute('width',bbox.width+6); bg.setAttribute('height',bbox.height+2);
        bg.setAttribute('fill','rgba(0,0,0,0.7)'); bg.setAttribute('rx','3');
        svg.insertBefore(bg, text);
    } catch(err) {}
};

window._clearRuler = window._clearRuler || function(containerId) {
    var container = document.getElementById(containerId);
    if (container && container._rulerSvg) {
        container._rulerSvg.innerHTML = '';
        container._rulerState = null;
    }
};
</script>"""

# ── Crop-rectangle JavaScript (SVG dashed-box selection) ─────────────────────
#
# Crop state is always stored in original image pixel coordinates so the box
# redraws correctly after browser window resize (no stale CSS-pixel values).
_CROP_JS = """<script>
window._initCrop = window._initCrop || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || container._cropInit) return;
    container._cropInit = true;
    container._cropMode = false;
    container._cropActive = false;
    container._cropRect = null;
    var svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:12;';
    container.appendChild(svg);
    container._cropSvg = svg;
    // Redraw overlay when container resizes (e.g. browser window resize)
    if (window.ResizeObserver) {
        var ro = new ResizeObserver(function() { window._drawCropRect(containerId); });
        ro.observe(container);
        container._cropRO = ro;
    }
};

// Convert original-image pixel (origX,origY) → SVG/screen position
window._cropPt = window._cropPt || function(container, origX, origY) {
    var zs  = container._zoomState || {scale:1,panX:0,panY:0};
    var meta = container._pixelMeta || {};
    var imgEl = container.querySelector('img');
    var cssW = imgEl ? imgEl.offsetWidth  : 0;
    var cssH = imgEl ? imgEl.offsetHeight : 0;
    var oW = meta.image_width  || cssW || 1;
    var oH = meta.image_height || cssH || 1;
    return {
        x: origX * (cssW / oW) * zs.scale + zs.panX,
        y: origY * (cssH / oH) * zs.scale + zs.panY
    };
};

// Convert SVG/screen position → original-image pixel
window._cropInv = window._cropInv || function(container, sx, sy) {
    var zs  = container._zoomState || {scale:1,panX:0,panY:0};
    var meta = container._pixelMeta || {};
    var imgEl = container.querySelector('img');
    var cssW = imgEl ? imgEl.offsetWidth  : 0;
    var cssH = imgEl ? imgEl.offsetHeight : 0;
    if (!cssW || !cssH) return {x:0, y:0};
    var oW = meta.image_width  || cssW;
    var oH = meta.image_height || cssH;
    return {
        x: (sx - zs.panX) / zs.scale * (oW / cssW),
        y: (sy - zs.panY) / zs.scale * (oH / cssH)
    };
};

window._setCropMode = window._setCropMode || function(containerId, enabled) {
    var container = document.getElementById(containerId);
    if (!container) return;
    if (!container._cropInit) window._initCrop(containerId);
    container._cropMode = enabled;
    var svg = container._cropSvg;
    svg.style.pointerEvents = enabled ? 'all' : 'none';
    container.style.cursor  = enabled ? 'crosshair' : '';
    if (enabled) {
        container._cropActive = true;

        container._cropMD = function(e) {
            e.preventDefault(); e.stopPropagation();
            var bRect = container.getBoundingClientRect();
            var sx = e.clientX - bRect.left, sy = e.clientY - bRect.top;
            var rs = container._cropRect;
            var mode = 'new';
            var THRESH = 12;
            if (rs) {
                var tl = window._cropPt(container, rs.origX1, rs.origY1);
                var tr = window._cropPt(container, rs.origX2, rs.origY1);
                var bl = window._cropPt(container, rs.origX1, rs.origY2);
                var br = window._cropPt(container, rs.origX2, rs.origY2);
                var near = function(p) {
                    return Math.abs(sx-p.x) < THRESH && Math.abs(sy-p.y) < THRESH;
                };
                if (near(tl) || near(tr))      mode = 'resize-top';
                else if (near(bl) || near(br)) mode = 'resize-bottom';
            }
            container._cropDragMode = mode;
            if (mode === 'new') {
                var meta = container._pixelMeta || {};
                var imgEl = container.querySelector('img');
                var cssW = imgEl ? imgEl.offsetWidth : 0;
                var oW = meta.image_width || cssW;
                var startOrig = window._cropInv(container, sx, sy);
                container._cropRect = {
                    origX1: 0, origY1: startOrig.y,
                    origX2: oW, origY2: startOrig.y,
                    origCenterX: oW / 2,
                    drawing: true
                };
            } else {
                rs.drawing = true;
            }
            window._drawCropRect(containerId);
        };

        container._cropMM = function(e) {
            var bRect = container.getBoundingClientRect();
            var sx = e.clientX - bRect.left, sy = e.clientY - bRect.top;
            var rs = container._cropRect;
            // While not dragging: update cursor near corners
            if (!rs || !rs.drawing) {
                if (rs) {
                    var THRESH2 = 12;
                    var tl2 = window._cropPt(container, rs.origX1, rs.origY1);
                    var tr2 = window._cropPt(container, rs.origX2, rs.origY1);
                    var bl2 = window._cropPt(container, rs.origX1, rs.origY2);
                    var br2 = window._cropPt(container, rs.origX2, rs.origY2);
                    var near2 = function(p) {
                        return Math.abs(sx-p.x) < THRESH2 && Math.abs(sy-p.y) < THRESH2;
                    };
                    container.style.cursor =
                        (near2(tl2)||near2(tr2)||near2(bl2)||near2(br2)) ? 'pointer' : 'crosshair';
                }
                return;
            }
            // Dragging: update crop rect
            e.preventDefault(); e.stopPropagation();
            var orig = window._cropInv(container, sx, sy);
            var cx = rs.origCenterX;
            var halfW = Math.abs(orig.x - cx);
            rs.origX1 = cx - halfW;
            rs.origX2 = cx + halfW;
            if (container._cropDragMode === 'resize-top') {
                rs.origY1 = orig.y;
            } else {
                rs.origY2 = orig.y;
            }
            window._drawCropRect(containerId);
        };

        container._cropMU = function(e) {
            var rs = container._cropRect;
            if (!rs) return;
            rs.drawing = false;
            window._drawCropRect(containerId);
            container.dispatchEvent(new CustomEvent('cropdrawn', {
                bubbles: true,
                detail: {
                    x1: Math.max(0, Math.round(rs.origX1)),
                    y1: Math.max(0, Math.round(Math.min(rs.origY1, rs.origY2))),
                    x2: Math.round(rs.origX2),
                    y2: Math.round(Math.max(rs.origY1, rs.origY2))
                }
            }));
        };

        svg.addEventListener('mousedown', container._cropMD);
        svg.addEventListener('mousemove', container._cropMM);
        svg.addEventListener('mouseup',   container._cropMU);
        window._drawCropRect(containerId);
    } else {
        container._cropActive = false;
        container.style.cursor = '';
        if (svg && container._cropMD) {
            svg.removeEventListener('mousedown', container._cropMD);
            svg.removeEventListener('mousemove', container._cropMM);
            svg.removeEventListener('mouseup',   container._cropMU);
        }
        if (svg) svg.innerHTML = '';
    }
};

window._drawCropRect = window._drawCropRect || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container || !container._cropSvg) return;
    if (!container._cropMode) return;
    var svg = container._cropSvg;
    var rs  = container._cropRect;
    svg.innerHTML = '';

    // Image dimensions (used for axis line + rect conversion)
    var meta  = container._pixelMeta || {};
    var imgEl = container.querySelector('img');
    var cssW  = imgEl ? imgEl.offsetWidth  : 0;
    var cssH  = imgEl ? imgEl.offsetHeight : 0;
    var oW = meta.image_width  || cssW || 1;
    var oH = meta.image_height || cssH || 1;

    // ── White dashed rotation-axis line (always in crop mode) ──
    var at = window._cropPt(container, oW / 2, 0);
    var ab = window._cropPt(container, oW / 2, oH);
    var axLine = document.createElementNS('http://www.w3.org/2000/svg','line');
    axLine.setAttribute('x1', at.x); axLine.setAttribute('y1', at.y);
    axLine.setAttribute('x2', ab.x); axLine.setAttribute('y2', ab.y);
    axLine.setAttribute('stroke', 'white');
    axLine.setAttribute('stroke-width', '2');
    axLine.setAttribute('stroke-dasharray', '8,4');
    axLine.setAttribute('opacity', '0.85');
    svg.appendChild(axLine);

    if (!rs) return;

    // ── Yellow dashed crop rectangle ──
    var p1 = window._cropPt(container, rs.origX1, rs.origY1);
    var p2 = window._cropPt(container, rs.origX2, rs.origY2);
    var rectEl = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rectEl.setAttribute('x', Math.min(p1.x,p2.x));
    rectEl.setAttribute('y', Math.min(p1.y,p2.y));
    rectEl.setAttribute('width',  Math.abs(p2.x-p1.x));
    rectEl.setAttribute('height', Math.abs(p2.y-p1.y));
    rectEl.setAttribute('fill',   'rgba(255,220,50,0.12)');
    rectEl.setAttribute('stroke', '#ffcc00');
    rectEl.setAttribute('stroke-width', '2');
    rectEl.setAttribute('stroke-dasharray', '8,4');
    svg.appendChild(rectEl);

    // ── Corner handles (draggable visual affordance) ──
    [
        [rs.origX1, rs.origY1], [rs.origX2, rs.origY1],
        [rs.origX1, rs.origY2], [rs.origX2, rs.origY2]
    ].forEach(function(coord) {
        var pt = window._cropPt(container, coord[0], coord[1]);
        var c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx', pt.x); c.setAttribute('cy', pt.y);
        c.setAttribute('r', '6');
        c.setAttribute('fill', '#ffcc00');
        c.setAttribute('stroke', '#997700');
        c.setAttribute('stroke-width', '1.5');
        svg.appendChild(c);
    });
};

window._clearCropRect = window._clearCropRect || function(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container._cropRect = null;
    window._drawCropRect(containerId);
    if (!container._cropMode && container._cropSvg) container._cropSvg.innerHTML = '';
    container.dispatchEvent(new CustomEvent('cropdrawn', {
        bubbles: true, detail: {x1:0, y1:0, x2:-1, y2:-1}
    }));
};
</script>"""


# ── Thread-pool helpers ───────────────────────────────────────────────────────

def _read_metadata(filepath: str, data_key: str = "data", elements_key: str = "elements", thetas_key: str = "thetas") -> dict:
    with h5py.File(filepath, "r") as f:
        shape = tuple(f[data_key].shape)   # (n_ch, n_ang, ny, nx)
        n_ch  = shape[0]

        if elements_key in f:
            elements = [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in f[elements_key][...]]
        elif "channels" in f:
            elements = [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in f["channels"][...]]
        else:
            elements = [f"Ch {i}" for i in range(n_ch)]

        thetas = f[thetas_key][...].tolist() if thetas_key in f else list(range(shape[1]))
        names  = (
            [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["names"][...]]
            if "names" in f else []
        )
        pixel_size_nm   = float(f["pixel_size_nm"][()]) if "pixel_size_nm" in f else None
        probe_energy_keV = float(f["probe_energy_keV"][()]) if "probe_energy_keV" in f else None
    return {"elements": elements, "thetas": thetas, "shape": shape, "names": names,
            "pixel_size_nm": pixel_size_nm, "probe_energy_keV": probe_energy_keV}


def _read_slice(filepath: str, ch_idx: int, ang_idx: int, data_key: str = "data") -> tuple:
    """Read one 2D slice and render to a Viridis PNG data URL."""
    with h5py.File(filepath, "r") as f:
        arr = np.array(f[data_key][ch_idx, ang_idx, :, :])
    vmin, vmax, vmean = float(arr.min()), float(arr.max()), float(arr.mean())
    ny, nx = arr.shape

    # Viridis PNG via plt.imsave — no Plotly, no extra deps
    p2  = float(np.nanpercentile(arr, 2))
    p98 = float(np.nanpercentile(arr, 98))
    if p98 <= p2:
        p98 = p2 + 1.0
    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap="viridis", vmin=p2, vmax=p98, format="png")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return data_url, vmin, vmax, vmean, ny, nx


# ── Component ─────────────────────────────────────────────────────────────────

def create_h5_inspector(
    fn_root_el,
    fn_data_el,
    on_elements_loaded=None,
    on_crop_changed=None,
    pixel_size_nm_ref=None,
    probe_energy_ref=None,
    data_key: str = "data",
    elements_key: str = "elements",
    thetas_key: str = "thetas",
) -> callable:
    """Inline HDF5 data inspector (load button + image viewer).

    Returns:
        Async callable ``load_file()`` — triggers the same logic as clicking
        "Load Data".  Call it programmatically to restore state after page
        navigation (file path must already be set on *fn_data_el*).

    Args:
        fn_root_el: NiceGUI input element holding the working directory path.
        fn_data_el: NiceGUI input element holding the HDF5 filename.
        on_elements_loaded: Optional callable(names: list[str]) called after file
            load with channel/element names.
        on_crop_changed: Optional callable(x1, y1, x2, y2) called when the user
            draws or clears a crop rectangle on the image. When None, crop UI is
            omitted entirely.
        pixel_size_nm_ref: Optional list[widget] — if provided, pixel_size_nm_ref[0]
            is a NiceGUI number widget whose value (nm/pixel) is injected into JS as
            container._pixelSizeNm so the ruler shows physical distance.
        probe_energy_ref: Optional list[widget] — if provided, probe_energy_ref[0]
            is a NiceGUI number widget auto-populated with probe_energy_keV from HDF5.
        data_key: HDF5 dataset path for the data array (default: "data" for BNL;
            use "exchange/data" for APS exchange format).
        elements_key: HDF5 dataset path for channel/element names (default: "elements";
            use "exchange/elements" for APS exchange format).
        thetas_key: HDF5 dataset path for rotation angles (default: "thetas";
            use "exchange/theta" for APS exchange format).
    """
    use_crop = on_crop_changed is not None

    ui.add_head_html(_ZOOM_JS)
    ui.add_head_html(_RULER_JS)
    if use_crop:
        ui.add_head_html(_CROP_JS)


    # ── Mutable state ──────────────────────────────────────────────
    meta       = {"loaded": False, "elements": [], "thetas": [], "shape": None}
    sel        = {"ch_idx": 0, "ang_idx": 0, "n_ang": 1}
    slider_ref   = {"el": None}
    last_change  = {"seq": 0}    # debounce sequence counter
    zoom_id      = {"cid": None}   # JS container id string set after DOM build

    # Helper: read cid without backslash inside f-string (pre-3.12 compat)
    def _cid() -> str:
        return zoom_id["cid"] or ""

    # Helper: return current pixel size (nm) from optional external widget ref
    def _get_pixel_size_nm() -> float:
        if pixel_size_nm_ref is not None and pixel_size_nm_ref[0] is not None:
            try:
                return float(pixel_size_nm_ref[0].value or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    # ── File load row ──────────────────────────────────────────────
    with ui.row().classes("w-full items-center gap-3"):
        load_btn   = ui.button("Load Data", icon="folder_open").props("dense outlined")
        status_lbl = ui.label("No file loaded.").classes("text-gray-400 text-sm flex-1")

    # ── Channel / angle selectors (hidden until loaded) ────────────
    controls = ui.row().classes("w-full items-center gap-4 flex-wrap")
    with controls:
        ch_select        = ui.select(label="Channel", options=[], value=None).classes("w-56").props("dense outlined")
        prev_btn         = ui.button(icon="chevron_left").props("dense flat")
        slider_container = ui.row().classes("flex-1 items-center")
        next_btn         = ui.button(icon="chevron_right").props("dense flat")
        angle_lbl        = ui.label("—").classes("text-sm text-gray-500 w-44 shrink-0")
    controls.set_visibility(False)

    # ── Image toolbar (zoom buttons + ruler + optional crop) ─────
    toolbar = ui.row().classes("w-full items-center gap-1 flex-wrap")
    with toolbar:
        ui.button(icon="zoom_out",   on_click=lambda: ui.run_javascript(f"window._zoomTo('{_cid()}', 1/1.4)")).props("round dense flat size=sm").classes("text-gray-500")
        ui.button(icon="fit_screen", on_click=lambda: ui.run_javascript(f"window._zoomReset('{_cid()}')")).props("round dense flat size=sm").classes("text-gray-500")
        ui.button(icon="zoom_in",    on_click=lambda: ui.run_javascript(f"window._zoomTo('{_cid()}', 1.4)")).props("round dense flat size=sm").classes("text-gray-500")
        ui.separator().props("vertical").style("height:20px; margin: 0 4px;")
        ruler_sw = ui.switch("Ruler", value=False).props("dense").classes("text-sm text-gray-500")
        ui.button("Clear", icon="clear", on_click=lambda: ui.run_javascript(f"window._clearRuler('{_cid()}')")).props("flat dense no-caps size=sm color=grey")
        if use_crop:
            ui.separator().props("vertical").style("height:20px; margin: 0 4px;")
            crop_sw = ui.switch("Crop", value=False).props("dense").classes("text-sm text-gray-500")
            ui.button("Clear Crop", icon="crop_free", on_click=lambda: ui.run_javascript(f"window._clearCropRect('{_cid()}')")).props("flat dense no-caps size=sm color=grey")
        else:
            crop_sw = None
    toolbar.set_visibility(False)

    # ── Crop readout label (shown below toolbar when crop is set) ───
    if use_crop:
        crop_lbl = ui.label("").classes("text-xs font-mono text-amber-700 w-full")
        crop_lbl.set_visibility(False)
    else:
        crop_lbl = None

    # ── Zoom container + image ─────────────────────────────────────
    # Plain <img> injected via JS (avoids q-img aspect-ratio complications).
    # width:100%; height:auto gives correct natural aspect ratio automatically.
    with ui.element("div").style("width:100%;max-width:600px;overflow:hidden;") as zoom_wrap:
        pass
    zoom_wrap.set_visibility(False)

    # ── Stats label ────────────────────────────────────────────────
    stats_lbl = ui.label("").classes("text-xs font-mono text-gray-500 text-center w-full")

    # Initialise zoom & ruler JS after DOM is ready.
    # Also injects a plain <img> into the zoom container — this gives correct
    # aspect ratio via width:100%/height:auto without any Quasar q-img overhead.
    async def _init_zoom_js():
        cid = f"c{zoom_wrap.id}"
        zoom_id["cid"] = cid
        await ui.run_javascript(
            f"(function(){{"
            f"var c=document.getElementById('{cid}');"
            f"if(!c)return;"
            f"var img=document.createElement('img');"
            f"img.style.cssText='width:100%;height:auto;display:block;';"
            f"c.appendChild(img);"
            f"}})()"
        )
        await ui.run_javascript(f"window._initZoomable('{cid}')")
        await ui.run_javascript(f"window._initRuler('{cid}')")
        if use_crop:
            await ui.run_javascript(f"window._initCrop('{cid}')")

    ui.timer(0.5, _init_zoom_js, once=True)

    # ── Helpers ────────────────────────────────────────────────────
    def _filepath() -> str:
        root = fn_root_el.value.strip()
        name = fn_data_el.value.strip()
        return name if os.path.isabs(name) else os.path.join(root, name)

    def _angle_text(idx: int) -> str:
        if meta["thetas"]:
            return f"{meta['thetas'][idx]:.2f}°  (idx {idx})"
        return f"idx {idx}"

    async def _show_slice():
        fpath   = _filepath()
        ch_idx  = sel["ch_idx"]
        ang_idx = sel["ang_idx"]
        ch_name = meta["elements"][ch_idx] if meta["elements"] else str(ch_idx)
        theta   = meta["thetas"][ang_idx]  if meta["thetas"]   else ang_idx
        try:
            data_url, vmin, vmax, vmean, ny, nx = await run.io_bound(
                _read_slice, fpath, ch_idx, ang_idx, data_key
            )
            # Preload the new image off-screen; swap the plain <img> src only
            # once fully decoded — no flash, no Quasar aspect-ratio complications.
            cid = _cid()
            if cid:
                await ui.run_javascript(
                    f"(function(){{"
                    f"var c=document.getElementById('{cid}');"
                    f"if(!c)return;"
                    f"var img=c.querySelector('img');"
                    f"if(!img)return;"
                    f"var tmp=new Image();"
                    f"tmp.onload=function(){{img.src=tmp.src;}};"
                    f"tmp.src='{data_url}';"
                    f"c._pixelMeta={{image_width:{nx},image_height:{ny}}};"
                    f"c._pixelSizeNm={_get_pixel_size_nm()};"
                    f"c.style.maxWidth=Math.max(256,Math.min(600,{nx}*5))+'px';"
                    f"}})()"
                )
            stats_lbl.set_text(
                f"{ch_name}  |  θ = {theta:.2f}°  |  "
                f"min {vmin:.4e}    max {vmax:.4e}    mean {vmean:.4e}"
                f"    {ny} × {nx} px"
            )
        except Exception as e:
            stats_lbl.set_text(f"Error: {e}")

    # ── Load file ──────────────────────────────────────────────────
    async def on_load_file():
        fpath = _filepath()
        if not fpath:
            ui.notify("Set Working Directory and HDF5 Data File first.", type="warning")
            return
        if not os.path.exists(fpath):
            ui.notify(f"File not found:\n{fpath}", type="negative")
            return

        load_btn.disable()
        status_lbl.set_text(f"Reading {os.path.basename(fpath)} …")
        status_lbl.classes(remove="text-green-600 text-red-500", add="text-gray-400")

        try:
            result = await run.io_bound(_read_metadata, fpath, data_key, elements_key, thetas_key)
            meta.update(result)
            meta["loaded"] = True

            # Auto-populate pixel size and beam energy if stored in the HDF5 file.
            # Must happen BEFORE on_elements_loaded so downstream code can read
            # the pixel size (e.g. to compute physical sample dimensions).
            if pixel_size_nm_ref is not None and pixel_size_nm_ref[0] is not None:
                pixel_size_nm_ref[0].set_value(result.get("pixel_size_nm"))
            if probe_energy_ref is not None and probe_energy_ref[0] is not None:
                probe_energy_ref[0].set_value(result.get("probe_energy_keV"))

            # Notify external listeners (e.g. parameter form element selector).
            # Pass the same list used by the channel dropdown for consistency.
            if on_elements_loaded is not None:
                on_elements_loaded(result["elements"], result["shape"])

            n_ch, n_ang, ny, nx = result["shape"]
            ch_select.options = result["elements"]
            ch_select.value   = result["elements"][0] if result["elements"] else None
            sel["ch_idx"]     = 0
            sel["ang_idx"]    = 0
            sel["n_ang"]      = n_ang

            # Recreate slider with correct max from the start
            slider_container.clear()
            with slider_container:
                sl = ui.slider(
                    min=0, max=n_ang - 1, step=1, value=0,
                    on_change=on_angle_change,
                ).classes("flex-1").props("label")
                slider_ref["el"] = sl

            angle_lbl.set_text(_angle_text(0))
            controls.set_visibility(True)
            toolbar.set_visibility(True)
            zoom_wrap.set_visibility(True)
            status_lbl.set_text(
                f"{os.path.basename(fpath)} — "
                f"{n_ch} channels, {n_ang} angles, {ny}×{nx} px"
            )
            status_lbl.classes(remove="text-gray-400 text-red-500", add="text-green-600")

            # Reset crop overlay and values when loading a new file
            if use_crop:
                ui.run_javascript(f"window._clearCropRect('{_cid()}')")
                if crop_sw is not None:
                    crop_sw.set_value(False)
                if crop_lbl is not None:
                    crop_lbl.set_text("")
                    crop_lbl.set_visibility(False)
                if on_crop_changed is not None:
                    on_crop_changed(0, 0, -1, -1)

            await _show_slice()

        except Exception as e:
            status_lbl.set_text(f"Error: {e}")
            status_lbl.classes(remove="text-gray-400 text-green-600", add="text-red-500")
        finally:
            load_btn.enable()

    # ── Channel change → immediate reload ──────────────────────────
    async def on_channel_change(e):
        if not meta["loaded"]:
            return
        try:
            sel["ch_idx"] = meta["elements"].index(e.value)
        except ValueError:
            sel["ch_idx"] = 0
        await _show_slice()

    # ── Angle slider — debounced to avoid concurrent IO ─────────────
    async def on_angle_change(e):
        if not meta["loaded"]:
            return
        try:
            idx = int(e.value)
        except (TypeError, ValueError):
            return
        sel["ang_idx"] = idx
        angle_lbl.set_text(_angle_text(idx))

        # Debounce: stay in the same coroutine (preserves NiceGUI client context).
        # Each event increments the sequence; only the last one actually loads.
        seq = last_change["seq"] + 1
        last_change["seq"] = seq
        await asyncio.sleep(0.25)
        if last_change["seq"] == seq:
            await _show_slice()

    # ── Prev / Next buttons ─────────────────────────────────────────
    async def go_to_angle(idx: int):
        if not meta["loaded"]:
            return
        idx = max(0, min(idx, sel["n_ang"] - 1))
        sel["ang_idx"] = idx
        if slider_ref["el"] is not None:
            slider_ref["el"].set_value(idx)
        angle_lbl.set_text(_angle_text(idx))
        await _show_slice()

    async def on_prev(_):
        await go_to_angle(sel["ang_idx"] - 1)

    async def on_next(_):
        await go_to_angle(sel["ang_idx"] + 1)

    # ── Ruler toggle (disables crop when active) ───────────────────
    def on_ruler_change(e):
        cid = _cid()
        if cid:
            ui.run_javascript(
                f"window._setRulerMode('{cid}', {str(e.value).lower()})"
            )
        if e.value:
            crop_sw.set_value(False)

    # ── Crop toggle (disables ruler when active) ────────────────────
    def on_crop_change(e):
        cid = _cid()
        if cid:
            ui.run_javascript(
                f"window._setCropMode('{cid}', {str(e.value).lower()})"
            )
        if e.value:
            ruler_sw.set_value(False)

    # ── Crop drawn / cleared event from JS ──────────────────────────
    def on_crop_drawn(e):
        data = e.args if isinstance(e.args, dict) else {}
        detail = data.get("detail", data)
        x1 = int(detail.get("x1", 0))
        y1 = int(detail.get("y1", 0))
        x2 = int(detail.get("x2", -1))
        y2 = int(detail.get("y2", -1))
        if crop_lbl is not None:
            if x2 < 0:
                crop_lbl.set_text("")
                crop_lbl.set_visibility(False)
            else:
                crop_lbl.set_text(
                    f"Crop region:  X [{x1} → {x2}]  Y [{y1} → {y2}]  px"
                )
                crop_lbl.set_visibility(True)
        if on_crop_changed is not None:
            on_crop_changed(x1, y1, x2, y2)

    # ── Wire events ────────────────────────────────────────────────
    load_btn.on_click(on_load_file)
    ch_select.on_value_change(on_channel_change)
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    ruler_sw.on_value_change(on_ruler_change)
    if crop_sw is not None:
        crop_sw.on_value_change(on_crop_change)
    if use_crop:
        zoom_wrap.on("cropdrawn", on_crop_drawn)

    return on_load_file
