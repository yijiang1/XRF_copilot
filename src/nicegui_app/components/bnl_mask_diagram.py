"""Interactive 3D BNL detector mask diagram using Plotly.

Shows sample cube, probe beam, rotation axis, and the pyramidal
detection solid-angle mask. Updates live when mask parameters change.
"""

import json
import math
import numpy as np
import plotly.graph_objects as go
from nicegui import ui


# Fixed schematic length for the pyramid (cm) — not physical, just display
_PYRAMID_L = 1.2
# Max half-width/height displayed (cap for large angles)
_MAX_HW = 1.8

# Fixed sample cube size
_SH = 0.25   # half-side
_SS = 0.50   # full side
_CZ = 0.25   # cube centre z


def _cube_mesh(cx, cy, cz, size, color="rgba(79,195,247,0.25)"):
    """Return a Mesh3d trace for a cube centred at (cx, cy, cz)."""
    h = size / 2
    x = [cx - h, cx + h, cx + h, cx - h, cx - h, cx + h, cx + h, cx - h]
    y = [cy - h, cy - h, cy + h, cy + h, cy - h, cy - h, cy + h, cy + h]
    z = [cz - h, cz - h, cz - h, cz - h, cz + h, cz + h, cz + h, cz + h]
    i = [0, 0, 0, 0, 4, 4, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 1, 4, 5, 6, 3, 6, 3, 4, 5, 2]
    k = [2, 3, 5, 5, 6, 7, 7, 7, 4, 3, 6, 6]
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.20, flatshading=True,
        hoverinfo="skip", showlegend=False,
    )


def create_bnl_mask_diagram(
    det_alfa: float = 20.6,
    det_theta: float = 20.6,
    mask_length_maximum: int = 200,
    width: int = 420,
    height: int = 380,
):
    """Create an interactive 3D BNL detector mask diagram.

    Returns ``update(det_alfa, det_theta, mask_length_maximum)`` to redraw
    on parameter change.
    """
    fig = go.Figure()
    plot_el = ui.plotly(fig).classes(
        "w-full border rounded-lg"
    ).style("border: 1px solid #e0e0e0; border-radius: 8px;")
    _state = {"first": True}

    def _build(alfa_deg, theta_deg, mask_len, sample_size_cm=0.0):
        # Pyramid half-angles (the code uses half-angles internally)
        alfa_half = math.radians(alfa_deg / 2.0)
        theta_half = math.radians(theta_deg / 2.0)

        # Half-width and half-height at pyramid face, capped for display
        hw = min(_PYRAMID_L * math.tan(alfa_half), _MAX_HW)
        hh = min(_PYRAMID_L * math.tan(theta_half), _MAX_HW)

        # Pyramid apex: at the +x face of the sample cube, centred
        apex_x = _SH
        face_x = _SH + _PYRAMID_L

        traces = []

        # ── Sample cube ───────────────────────────────────────────
        traces.append(_cube_mesh(0, 0, _CZ, _SS, "rgba(79,195,247,0.25)"))

        # Sample wireframe edges
        ex = [-_SH, _SH, _SH, -_SH, -_SH, None, -_SH, _SH, _SH, -_SH, -_SH, None,
              -_SH, -_SH, None, _SH, _SH, None, _SH, _SH, None, -_SH, -_SH]
        ey = [-_SH, -_SH, _SH, _SH, -_SH, None, -_SH, -_SH, _SH, _SH, -_SH, None,
              -_SH, -_SH, None, -_SH, -_SH, None, _SH, _SH, None, _SH, _SH]
        ez = [0, 0, 0, 0, 0, None,
              _SS, _SS, _SS, _SS, _SS, None,
              0, _SS, None, 0, _SS, None,
              0, _SS, None, 0, _SS]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#0277BD", width=2),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Rotation axis (z) ─────────────────────────────────────
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-0.15, _SS + 0.25],
            mode="lines", line=dict(color="#000000", width=3, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Beam arrow (along +y) ─────────────────────────────────
        b_start = -_SH - 0.5
        b_end = _SH + 0.2
        traces.append(go.Scatter3d(
            x=[0, 0], y=[b_start, b_end], z=[_CZ, _CZ],
            mode="lines", line=dict(color="#E65100", width=6),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Cone(
            x=[0], y=[b_end], z=[_CZ],
            u=[0], v=[0.1], w=[0],
            sizemode="absolute", sizeref=0.08,
            colorscale=[[0, "#E65100"], [1, "#E65100"]],
            showscale=False, hoverinfo="skip", showlegend=False,
        ))

        # ── Detection pyramid (semi-transparent mesh) ─────────────
        # 5 vertices: apex + 4 corners
        # Corners: C1 (bottom-left), C2 (bottom-right), C3 (top-right), C4 (top-left)
        # y is horizontal (along beam direction), z is vertical (rotation axis)
        px = [apex_x, face_x, face_x, face_x, face_x]
        py = [0.0,   -hw,    hw,    hw,   -hw]
        pz = [_CZ,  _CZ-hh, _CZ-hh, _CZ+hh, _CZ+hh]
        # 4 triangular side faces: apex(0) to each edge
        pi = [0, 0, 0, 0]
        pj = [1, 2, 3, 4]
        pk = [2, 3, 4, 1]
        traces.append(go.Mesh3d(
            x=px, y=py, z=pz,
            i=pi, j=pj, k=pk,
            color="#66BB6A", opacity=0.20, flatshading=True,
            hoverinfo="skip", showlegend=False,
        ))

        # ── Pyramid wireframe ─────────────────────────────────────
        # 4 edges from apex to corners
        wire_x = [apex_x, face_x, None, apex_x, face_x, None,
                  apex_x, face_x, None, apex_x, face_x, None,
                  # Rectangle at face
                  face_x, face_x, face_x, face_x, face_x]
        wire_y = [-hw if v else 0 for v in [False, True, None,
                                             False, True, None,
                                             False, True, None,
                                             False, True, None]]
        # Build wireframe properly
        wx = []
        wy = []
        wz = []
        # Apex to each corner
        corners_y = [-hw, hw, hw, -hw]
        corners_z = [_CZ - hh, _CZ - hh, _CZ + hh, _CZ + hh]
        for cy_, cz_ in zip(corners_y, corners_z):
            wx += [apex_x, face_x, None]
            wy += [0, cy_, None]
            wz += [_CZ, cz_, None]
        # Rectangle face
        rect_y = [-hw, hw, hw, -hw, -hw]
        rect_z = [_CZ - hh, _CZ - hh, _CZ + hh, _CZ + hh, _CZ - hh]
        wx += [face_x] * 5
        wy += rect_y
        wz += rect_z
        traces.append(go.Scatter3d(
            x=wx, y=wy, z=wz, mode="lines",
            line=dict(color="#2E7D32", width=2),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Angle arc (horizontal, showing alfa) ──────────────────
        # Arc in xz-plane (y=0) sweeping the horizontal half-angle
        n_arc = 20
        arc_r = _PYRAMID_L * 0.45
        arc_angles = np.linspace(-alfa_half, alfa_half, n_arc)
        arc_x = apex_x + arc_r * np.cos(arc_angles)
        arc_y = arc_r * np.sin(arc_angles)
        arc_z = np.full(n_arc, _CZ)
        traces.append(go.Scatter3d(
            x=arc_x, y=arc_y, z=arc_z, mode="lines",
            line=dict(color="#388E3C", width=2),
            hoverinfo="skip", showlegend=False,
        ))

        # ── Labels (scene annotations — fixed pixel size) ─────────
        sample_label = "Sample"
        if sample_size_cm > 0:
            sample_label = f"Sample ({sample_size_cm * 1e4:.1f} \u03bcm)"

        annotations = [
            dict(x=0, y=0, z=_SS + 0.18, text=sample_label, showarrow=False,
                 font=dict(size=13, color="#0277BD")),
            dict(x=0.04, y=0, z=_SS + 0.30, text="z (rotation)", showarrow=False,
                 font=dict(size=11, color="#000000")),
            dict(x=0, y=b_start - 0.06, z=_CZ + 0.06, text="Beam", showarrow=False,
                 font=dict(size=11, color="#E65100")),
            dict(x=face_x + 0.05, y=0, z=_CZ + hh + 0.12,
                 text="Detection Region", showarrow=False,
                 font=dict(size=12, color="#2E7D32")),
            dict(x=face_x + 0.05, y=hw + 0.04, z=_CZ,
                 text=f"\u03b1 {alfa_deg:.1f}\u00b0", showarrow=False,
                 font=dict(size=13, color="#388E3C")),
            dict(x=face_x + 0.05, y=0, z=_CZ - hh - 0.10,
                 text=f"\u03b8 {theta_deg:.1f}\u00b0", showarrow=False,
                 font=dict(size=13, color="#388E3C")),
        ]

        # ── Layout ────────────────────────────────────────────────
        max_x = face_x + 0.4
        max_yz = max(hw, _SH) + 0.5
        span = max(max_x, max_yz, _SH + 0.8)

        camera = dict(
            eye=dict(x=0.48, y=-0.6, z=0.36),
            center=dict(x=0.08, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        )

        layout = go.Layout(
            width=width, height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                annotations=annotations,
                xaxis=dict(visible=False, range=[-span, span]),
                yaxis=dict(visible=False, range=[-span, span]),
                zaxis=dict(visible=False, range=[_CZ - span, _CZ + span]),
                aspectmode="cube",
                camera=camera,
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return traces, layout

    def _render(alfa_deg, theta_deg, mask_len, sample_size_cm=0.0):
        traces, layout = _build(alfa_deg, theta_deg, mask_len, sample_size_cm)
        new_fig = go.Figure(data=traces, layout=layout)
        if _state["first"]:
            plot_el.update_figure(new_fig)
            _state["first"] = False
        else:
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

    _render(det_alfa, det_theta, mask_length_maximum)

    def update(det_alfa, det_theta, mask_length_maximum, sample_size_cm=0.0):
        _render(det_alfa, det_theta, mask_length_maximum, sample_size_cm)

    return update
