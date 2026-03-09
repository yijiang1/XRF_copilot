"""Interactive 3D detector geometry diagram using Plotly.

Shows sample cube, probe beam, rotation axis, and circular detector
with pixel grid. Updates live when detector parameters change.
"""

import json
import math
import numpy as np
import plotly.graph_objects as go
from nicegui import ui


def _cube_mesh(cx, cy, cz, size, color="rgba(79,195,247,0.25)"):
    """Return a Mesh3d trace for a cube centred at (cx, cy, cz)."""
    h = size / 2
    # 8 vertices
    x = [cx - h, cx + h, cx + h, cx - h, cx - h, cx + h, cx + h, cx - h]
    y = [cy - h, cy - h, cy + h, cy + h, cy - h, cy - h, cy + h, cy + h]
    z = [cz - h, cz - h, cz - h, cz - h, cz + h, cz + h, cz + h, cz + h]
    # 12 triangles (2 per face)
    i = [0, 0, 0, 0, 4, 4, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 1, 4, 5, 6, 3, 6, 3, 4, 5, 2]
    k = [2, 3, 5, 5, 6, 7, 7, 7, 4, 3, 6, 6]
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.20, flatshading=True,
        hoverinfo="skip", showlegend=False,
    )


def _circle_pts(radius, n=48):
    """Return (y_arr, z_arr) tracing a circle centred at origin."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    return radius * np.cos(a), radius * np.sin(a)


def create_detector_diagram(
    det_dia_cm: float = 0.9,
    det_from_sample_cm: float = 1.6,
    det_ds_spacing_cm: float = 0.4,
    det_on_which_side: str = "positive",
    width: int = 420,
    height: int = 380,
):
    """Create an interactive 3D detector geometry diagram.

    Returns ``update(dia, dist, spacing, side, sample_size_cm)`` to redraw
    on param change.  ``sample_size_cm=0`` uses a default representative size.
    """

    # Fixed sample cube size (0.5 cm side) — never rescales
    SH = 0.25       # half-side of sample cube
    SS = 0.50       # full side
    CZ = 0.25       # cube centre z = SH (cube sits on z=0)

    fig = go.Figure()
    plot_el = ui.plotly(fig).classes(
        "w-full border rounded-lg"
    ).style("border: 1px solid #e0e0e0; border-radius: 8px;")
    _state = {"first": True}

    def _build(dia, dist, spacing, side, sample_size_cm=0.0):
        sign = 1.0 if side == "positive" else -1.0

        # Use physical cm values directly — no auto-scaling
        det_r = max(dia / 2, 0.015)
        det_x = sign * (SH + dist)

        traces = []

        # ── Sample cube ───────────────────────────────────────────
        traces.append(_cube_mesh(0, 0, CZ, SS, "rgba(79,195,247,0.25)"))

        # Sample wireframe edges
        ex = [-SH, SH, SH, -SH, -SH, None, -SH, SH, SH, -SH, -SH, None,
              -SH, -SH, None, SH, SH, None, SH, SH, None, -SH, -SH]
        ey = [-SH, -SH, SH, SH, -SH, None, -SH, -SH, SH, SH, -SH, None,
              -SH, -SH, None, -SH, -SH, None, SH, SH, None, SH, SH]
        ez = [0, 0, 0, 0, 0, None,
              SS, SS, SS, SS, SS, None,
              0, SS, None, 0, SS, None,
              0, SS, None, 0, SS]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#0277BD", width=2),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Rotation axis (z) ─────────────────────────────────────
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-0.15, SS + 0.25],
            mode="lines", line=dict(color="#000000", width=3, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Beam arrow (along +y) ─────────────────────────────────
        b_start = -SH - 0.5
        b_end = SH + 0.2
        traces.append(go.Scatter3d(
            x=[0, 0], y=[b_start, b_end], z=[CZ, CZ],
            mode="lines", line=dict(color="#E65100", width=6),
            hoverinfo="skip", showlegend=False,
        ))
        # arrowhead as a cone
        traces.append(go.Cone(
            x=[0], y=[b_end], z=[CZ],
            u=[0], v=[0.1], w=[0],
            sizemode="absolute", sizeref=0.08,
            colorscale=[[0, "#E65100"], [1, "#E65100"]],
            showscale=False, hoverinfo="skip", showlegend=False,
        ))

        # ── Detector circle ───────────────────────────────────────
        cy, cz_circ = _circle_pts(det_r)
        det_x_arr = np.full_like(cy, det_x)
        traces.append(go.Scatter3d(
            x=det_x_arr, y=cy, z=cz_circ + CZ,
            mode="lines", line=dict(color="#2E7D32", width=3),
            hoverinfo="skip", showlegend=False,
        ))
        # cross-hairs
        traces.append(go.Scatter3d(
            x=[det_x, det_x, None, det_x, det_x],
            y=[-det_r, det_r, None, 0, 0],
            z=[CZ, CZ, None, CZ - det_r, CZ + det_r],
            mode="lines", line=dict(color="#66BB6A", width=1),
            hoverinfo="skip", showlegend=False,
        ))

        # detector pixel dots
        if spacing > 0 and dia > 0:
            n_pts = min(max(1, int(dia / spacing) + 1), 9)
            r_phys = dia / 2
            coords = np.linspace(-r_phys, r_phys, n_pts)
            yy, zz = np.meshgrid(coords, coords)
            mask = yy.ravel() ** 2 + zz.ravel() ** 2 <= r_phys ** 2
            if mask.any():
                traces.append(go.Scatter3d(
                    x=np.full(mask.sum(), det_x),
                    y=yy.ravel()[mask],
                    z=zz.ravel()[mask] + CZ,
                    mode="markers",
                    marker=dict(size=3, color="#1B5E20"),
                    hoverinfo="skip", showlegend=False,
                ))

        # ── Distance annotation line ──────────────────────────────
        edge_x = sign * SH
        az = -0.06
        traces.append(go.Scatter3d(
            x=[edge_x, det_x], y=[0, 0], z=[az, az],
            mode="lines", line=dict(color="#9E9E9E", width=2),
            hoverinfo="skip", showlegend=False,
        ))
        # tick marks
        traces.append(go.Scatter3d(
            x=[edge_x, edge_x, None, det_x, det_x],
            y=[0, 0, None, 0, 0],
            z=[az - 0.03, az + 0.03, None, az - 0.03, az + 0.03],
            mode="lines", line=dict(color="#9E9E9E", width=2),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Labels (as Scatter3d text) ────────────────────────────
        sample_label = "Sample"
        if sample_size_cm > 0:
            sample_label = f"Sample ({sample_size_cm * 1e4:.1f} \u03bcm)"

        # ── Labels as scene annotations (2D overlays at 3D positions) ─
        # These render at fixed pixel size regardless of zoom or scene scale.
        annotations = [
            dict(x=0, y=0, z=SS + 0.18, text=sample_label, showarrow=False,
                 font=dict(size=13, color="#0277BD")),
            dict(x=0.04, y=0, z=SS + 0.30, text="z (rotation)", showarrow=False,
                 font=dict(size=11, color="#000000")),
            dict(x=0, y=b_start - 0.06, z=CZ + 0.06, text="Beam", showarrow=False,
                 font=dict(size=11, color="#E65100")),
            dict(x=det_x, y=0, z=CZ + det_r + 0.08, text="Detector", showarrow=False,
                 font=dict(size=12, color="#2E7D32")),
            dict(x=(edge_x + det_x) / 2, y=0, z=az - 0.10,
                 text=f"{dist:.1f} cm", showarrow=False,
                 font=dict(size=13, color="#616161")),
            dict(x=det_x, y=det_r + 0.04, z=CZ - det_r - 0.06,
                 text=f"D {dia:.1f} cm", showarrow=False,
                 font=dict(size=13, color="#388E3C")),
        ]

        # ── Layout ────────────────────────────────────────────────
        # Dynamic axis ranges that fit the content snugly.
        # Labels use scene.annotations (2D overlays) so their pixel
        # size is constant regardless of axis range changes.
        max_x = abs(det_x) + det_r + 0.3
        max_yz = max(det_r, SH) + 0.6
        span = max(max_x, max_yz, SH + 0.8)

        cam_sign = sign

        camera = dict(
            eye=dict(x=cam_sign * 0.48, y=-0.6, z=0.36),
            center=dict(x=cam_sign * 0.08, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        )

        layout = go.Layout(
            width=width, height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                annotations=annotations,
                xaxis=dict(visible=False, range=[-span, span]),
                yaxis=dict(visible=False, range=[-span, span]),
                zaxis=dict(visible=False, range=[CZ - span, CZ + span]),
                aspectmode="cube",
                camera=camera,
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return traces, layout

    def _render(dia, dist, spacing, side, sample_size_cm=0.0):
        traces, layout = _build(dia, dist, spacing, side, sample_size_cm)
        new_fig = go.Figure(data=traces, layout=layout)
        if _state["first"]:
            # Initial render — include camera to set default view
            plot_el.update_figure(new_fig)
            _state["first"] = False
        else:
            # Subsequent renders — use JS to call Plotly.react directly.
            # Inject the current camera into the new layout BEFORE react
            # so the view never flickers.
            fig_dict = new_fig.to_plotly_json()
            fig_dict.get("layout", {}).get("scene", {}).pop("camera", None)
            fig_json = json.dumps(fig_dict, default=str)
            js = f"""
            (() => {{
                const comp = getElement({plot_el.id});
                if (!comp) return;
                const gd = comp.$el;
                if (!gd || !gd._fullLayout) return;
                const fig = {fig_json};
                const sc = gd._fullLayout.scene;
                if (sc && sc.camera) {{
                    fig.layout.scene = fig.layout.scene || {{}};
                    fig.layout.scene.camera = JSON.parse(JSON.stringify(sc.camera));
                }}
                Plotly.react(gd, fig.data, fig.layout);
            }})();
            """
            ui.run_javascript(js)

    _render(det_dia_cm, det_from_sample_cm, det_ds_spacing_cm, det_on_which_side)

    def update(det_dia_cm, det_from_sample_cm, det_ds_spacing_cm,
               det_on_which_side, sample_size_cm=0.0):
        _render(det_dia_cm, det_from_sample_cm, det_ds_spacing_cm,
                det_on_which_side, sample_size_cm)

    return update
