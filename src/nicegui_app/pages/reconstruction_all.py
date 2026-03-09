"""Unified reconstruction page — all three methods on one page.

Layout:
  Backend connection row
  [Card]      Shared fn_root / fn_data inputs
  [Tabs]      BNL | Panpan | Wendy
    BNL tab:    H5 inspector + element selector + BNL params
    Panpan tab: H5 inspector (exchange format) + Panpan params
    Di tab:     H5 inspector (exchange format) + Di params
  [Shared]    Run/Stop + Progress (shows active tab's method)
  [Shared]    Status log (combines messages from all three methods)
"""

import asyncio
import os
import time
import httpx
from urllib.parse import urlparse
from nicegui import ui
from ..state import FLState, ReconState, AppState
from ..api_client import XRFSimulationAPIClient
from ..components.fl_parameter_form import (
    _ElementSelector,
    _ICSelector,
    _IndicesProxy,
    _ShellProxy,
    _DensityProxy,
    _EmissionProxy,
    _ValueHolder,
)
from ..components.h5_inspector import create_h5_inspector
from ..components.file_browser import open_file_browser, BrowseConfig, BrowseMode
from ..components.fl_control_panel import _collect_fl_params
from ..components.recon_control_panel import _collect_recon_params
from ..components.di_control_panel import _collect_di_params
from ..components.status_log import create_status_log
from ..components.session_bar import create_session_bar
from ..components.result_viewer import create_result_viewer
from ..components.detector_diagram import create_detector_diagram
from ..utils.message_formatter import append_to_message_list

# Colour legend for the periodic-table element selector
_LEGEND = [
    ("#aacce8", "Transition metals"),
    ("#f4a8a8", "Alkali / Alkaline-earth"),
    ("#a8d4c8", "Lanthanides"),
    ("#a8d8b0", "Post-transition metals"),
    ("#f5e47a", "Metalloids"),
    ("#d4eaa8", "Nonmetals"),
    ("#d4c8e8", "Noble gases"),
    ("#b8b8c8", "Other / TFY / IC"),
]


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 8000)
    return host, port


# ── Module-level persistence (survives page navigations) ─────────────────────
_saved_params: dict = {}
_saved_tab: dict = {"value": "BNL"}
_saved_session_state: dict = {
    "selected_id": "",
    "fl_messages": [],
    "recon_messages": [],
    "di_messages": [],
    "fl_log_offset": 0,
    "recon_log_offset": 0,
    "di_log_offset": 0,
    "fl_err_reported": False,
    "recon_err_reported": False,
    "di_err_reported": False,
}

# Keys in the per-method input dicts that are real NiceGUI form widgets
_FL_FORM_KEYS = frozenset({
    "binning_factor", "scale", "det_alfa", "det_theta", "mask_length_maximum",
    "recon_method", "recon_n_iter", "n_correction_iters", "correction_n_iter",
    "border_pixels", "smooth_filter_size",
})
_RECON_FORM_KEYS = frozenset({
    "probe_intensity", "probe_att", "n_epochs", "save_every_n_epochs",
    "minibatch_size", "lr", "b1", "b2", "selfAb", "cont_from_check_point",
    "use_saved_initial_guess", "ini_kind", "init_const", "ini_rand_amp",
    "manual_det_area", "det_dia_cm", "det_from_sample_cm", "det_ds_spacing_cm",
    "det_on_which_side",
})
_DI_FORM_KEYS = frozenset({
    "probe_intensity", "probe_att", "n_outer_epochs", "lbfgs_n_iter",
    "lbfgs_history", "loss_type", "beta1_xrt", "tikhonov_lambda",
    "save_every_n_epochs", "minibatch_size", "selfAb", "cont_from_check_point",
    "use_saved_initial_guess", "ini_kind", "init_const",
    "manual_det_area", "det_dia_cm", "det_from_sample_cm", "det_ds_spacing_cm",
    "det_on_which_side",
})


def create_reconstruction_all_page(api_key: str = ""):
    """Build the unified reconstruction page (all three methods)."""

    api = XRFSimulationAPIClient(api_key=api_key)
    fl_state = FLState()
    recon_state = ReconState()
    di_state = ReconState()
    log_state = AppState()   # messages field used for combined status log

    backend_connected = {"value": False, "error_logged": False}
    init_host, init_port = _parse_endpoint(api.endpoint)

    # ── Per-method error / log-offset tracking ─────────────────────────────────
    fl_err    = {"reported": False, "log_offset": 0}
    recon_err = {"reported": False, "log_offset": 0}
    di_err    = {"reported": False, "log_offset": 0}

    with ui.column().classes("w-full px-4 gap-4"):

        # ── Backend connection row ────────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                ui.icon("dns", size="xs").classes("text-gray-400 self-center")
                ui.label("Backend").classes(
                    "text-sm font-semibold text-gray-500 self-center"
                )
                host_input = ui.input(
                    "Host", value=init_host
                ).classes("w-64").props("dense outlined")
                port_input = ui.input(
                    "Port", value=init_port
                ).classes("w-24").props("dense outlined")
                api_key_input = ui.input(
                    "API Key", value=api_key, password=True, password_toggle_button=True,
                ).classes("w-48").props("dense outlined")
                connect_btn = ui.button(
                    "Connect", icon="link"
                ).props("unelevated no-caps color=primary")
                ui.space()
                connection_label = ui.label(
                    f"{api.endpoint}"
                ).classes("text-xs text-gray-400 self-center")

        def on_connect():
            host = host_input.value.strip()
            port = port_input.value.strip()
            new_endpoint = f"http://{host}:{port}"
            api.set_endpoint(new_endpoint)
            api.set_api_key(api_key_input.value.strip())
            backend_connected["value"] = False
            backend_connected["error_logged"] = False
            connection_label.set_text(f"{new_endpoint} (connecting...)")
            connection_label.classes(
                remove="text-green-600 text-red-500 text-orange-500",
                add="text-gray-400",
            )

        connect_btn.on_click(on_connect)

        # ══════════════════════════════════════════════════════════════════════
        # SHARED FILE INPUTS (above tabs)
        # ══════════════════════════════════════════════════════════════════════
        # ── Proxies that split the single full-path input into dir + basename ──
        class _DirProxy:
            """Returns dirname of the full H5 path."""
            def __init__(self, el): self._el = el
            @property
            def value(self) -> str:
                p = self._el.value.strip()
                return os.path.dirname(p) if p else ""

        class _BasenameProxy:
            """Returns basename of the full H5 path."""
            def __init__(self, el): self._el = el
            @property
            def value(self) -> str:
                p = self._el.value.strip()
                return os.path.basename(p) if p else ""

        with ui.card().classes("w-full"):
            ui.label("Data").classes("text-base font-semibold px-3 pt-2")
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4 items-end"):
                    h5_path_el = ui.input(
                        "HDF5 Data File (full path)",
                        value="/mnt/micdata3/XRF_tomography/testing_ground/data/fl_correction/bnl_test.h5",
                        placeholder="/path/to/data.h5",
                    ).classes("flex-1 font-mono")
                    h5_path_el.tooltip(
                        "Full path to the HDF5 data file — shared by all three methods. "
                        "Results will be saved in the same folder. "
                        "Required keys: 'data' (n_ch, n_ang, H, W), 'elements' (channel names), 'rot_angles' (degrees)."
                    )

                    async def _browse_h5():
                        await open_file_browser(
                            BrowseConfig(
                                mode=BrowseMode.OPEN_FILE,
                                title="Select HDF5 File",
                                icon="folder_open",
                                file_extensions={".h5", ".hdf5"},
                                home_path="/mnt/micdata3/XRF_tomography",
                            ),
                            target_input=h5_path_el,
                        )

                    ui.button(icon="folder_open", on_click=_browse_h5).props(
                        "flat dense"
                    ).tooltip("Browse for HDF5 file")

                # ── Derive fn_root / fn_data proxies from the single path input ─
                fn_root_el = _DirProxy(h5_path_el)
                fn_data_el = _BasenameProxy(h5_path_el)

                # ── Shared H5 inspector (Load Data + data viewer) ─────────────
                # Placed before energy/pixel_size so that auto-loaded values
                # from HDF5 fill those fields after the user clicks "Load Data".
                elem_selector   = _ElementSelector()
                ic_selector     = _ICSelector()
                # Channel selectors for Panpan/Di data indexing
                recon_xrt_sel   = _ICSelector(default_name="abs_ic")
                recon_us_ic_sel = _ICSelector(default_name="us_ic")
                recon_ds_ic_sel = _ICSelector(default_name="ds_ic")
                di_xrt_sel      = _ICSelector(default_name="abs_ic")
                di_us_ic_sel    = _ICSelector(default_name="us_ic")
                di_ds_ic_sel    = _ICSelector(default_name="ds_ic")
                _elem_ui        = {"container": None, "status": None}
                _crop_cb_ref    = [None]

                # Forward-declare refs; actual widgets created below after
                # create_h5_inspector so that the Load button sits above them.
                _pixel_size_ref  = [None]
                _energy_ref      = [None]
                _data_shape      = {"nx": 0, "ny": 0}    # filled on HDF5 load
                _det_diagram_refs = []  # list of (update_fn, inputs_dict) for diagrams

                def on_elements_loaded(names: list, shape=None) -> None:
                    ctr = _elem_ui["container"]
                    st  = _elem_ui["status"]
                    if ctr is None:
                        return
                    elem_selector.populate(names, ctr)
                    ic_selector.populate(names)
                    # Populate channel selectors for Panpan/Di data indexing
                    for sel in (recon_xrt_sel, recon_us_ic_sel, recon_ds_ic_sel,
                                di_xrt_sel, di_us_ic_sel, di_ds_ic_sel):
                        sel.populate(names)
                    elem_selector.autofill_from_xraylib()
                    if st is not None:
                        st.set_text(
                            f"{len(names)} channel(s) detected — "
                            "toggle tiles to select elements for reconstruction:"
                        )
                        st.classes(remove="text-gray-400 italic", add="text-gray-600")
                    # Store data shape and refresh detector diagrams
                    if shape is not None and len(shape) >= 4:
                        _data_shape["ny"] = shape[2]
                        _data_shape["nx"] = shape[3]
                        _refresh_all_diagrams()

                def _compute_sample_size_cm():
                    """Compute physical sample width in cm from pixel_size_nm × nx."""
                    pix = (_pixel_size_ref[0].value
                           if _pixel_size_ref[0] is not None else None)
                    nx = _data_shape.get("nx", 0)
                    if pix and nx:
                        return nx * pix * 1e-7  # nm → cm
                    return 0.0

                def _refresh_all_diagrams():
                    """Refresh every registered detector diagram."""
                    scm = _compute_sample_size_cm()
                    for upd_fn, inp in _det_diagram_refs:
                        upd_fn(
                            det_dia_cm=inp["det_dia_cm"].value or 0.9,
                            det_from_sample_cm=inp["det_from_sample_cm"].value or 1.6,
                            det_ds_spacing_cm=inp["det_ds_spacing_cm"].value or 0.4,
                            det_on_which_side=inp["det_on_which_side"].value or "positive",
                            sample_size_cm=scm,
                        )

                _h5_load_file = create_h5_inspector(
                    _ValueHolder(""),
                    h5_path_el,
                    on_elements_loaded=on_elements_loaded,
                    on_crop_changed=lambda x1, y1, x2, y2: (
                        _crop_cb_ref[0](x1, y1, x2, y2) if _crop_cb_ref[0] else None
                    ),
                    pixel_size_nm_ref=_pixel_size_ref,
                    probe_energy_ref=_energy_ref,
                )

                # ── Beam energy & pixel size (below Load Data) ────────────────
                with ui.row().classes("w-full gap-4 items-end"):
                    probe_energy_el = ui.number(
                        "Beam Energy (keV)", value=None, step=0.01, min=0.1,
                        format="%.3f",
                        validation={"Required": lambda v: v is not None},
                    ).classes("w-48")
                    probe_energy_el.tooltip(
                        "Incident beam energy in keV — used by all three methods. "
                        "Auto-loaded from HDF5 if the file stores 'probe_energy_keV'."
                    )

                    pixel_size_nm_el = ui.number(
                        "Pixel Size (nm)", value=None, step=10, min=1,
                        validation={"Required": lambda v: v is not None},
                    ).classes("w-48")
                    pixel_size_nm_el.tooltip(
                        "Voxel size in nm — used by all three methods. "
                        "Auto-loaded from HDF5 if the file stores 'pixel_size_nm'. "
                        "For Panpan/Di, grid dimensions are auto-detected from data shape."
                    )

                # Wire up the refs now that the widgets exist
                _pixel_size_ref[0] = pixel_size_nm_el
                _energy_ref[0] = probe_energy_el

                # ── Element selector (shared — all three methods) ─────────────
                ui.separator().classes("mt-1")
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.label("Elements").classes("text-sm font-medium text-gray-700")
                    ui.space()
                    ui.button(
                        "All", on_click=lambda: elem_selector.select_all(True)
                    ).props("dense flat no-caps size=xs color=primary")
                    ui.button(
                        "None", on_click=lambda: elem_selector.select_all(False)
                    ).props("dense flat no-caps size=xs color=grey")

                status_lbl = ui.label(
                    "Load HDF5 file above to auto-detect element names."
                ).classes("text-xs text-gray-400 italic w-full")
                _elem_ui["status"] = status_lbl

                with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-0.5 mb-1"):
                    for _bg, _lbl in _LEGEND:
                        with ui.row().classes("items-center gap-1"):
                            ui.element("div").style(
                                f"width:11px;height:11px;background:{_bg};"
                                "border-radius:2px;border:1px solid rgba(0,0,0,0.15);"
                                "flex-shrink:0;"
                            )
                            ui.label(_lbl).classes("text-xs text-gray-500")

                elem_ctr = ui.row().classes("w-full flex-wrap gap-4 py-1")
                _elem_ui["container"] = elem_ctr

        # ══════════════════════════════════════════════════════════════════════
        # METHOD TABS
        # ══════════════════════════════════════════════════════════════════════
        with ui.tabs().classes("w-full").props("dense inline-label") as tabs:
            bnl_tab    = ui.tab("BNL")
            panpan_tab = ui.tab("Panpan")
            di_tab     = ui.tab("Wendy")

        with ui.tab_panels(tabs, value=bnl_tab).classes("w-full").style("padding: 0;"):

            # ══════════════════════════════════════════════════════════════════
            # BNL TAB
            # ══════════════════════════════════════════════════════════════════
            with ui.tab_panel(bnl_tab):
                with ui.column().classes("w-full gap-2 p-2"):

                    ui.label(
                        "Element tiles and shell selection are in the Data section above. "
                        "Density (ρ) and emission energy (E) within each tile are auto-filled "
                        "from xraylib and can be overridden manually."
                    ).classes("text-xs text-gray-500 italic mb-1")

                    # ── Ion-chamber selector ──────────────────────────────────
                    with ui.row().classes("w-full gap-4 items-end"):
                        ic_selector.create_widget("Ion Chamber Channel (BNL)").tooltip(
                            "Channel used for incident flux normalisation (BNL method only). "
                            "Populated automatically from the HDF5 file."
                        )

                    # ── Crop region (BNL only, set via interactive crop tool) ─
                    crop_x1 = _ValueHolder(0)
                    crop_y1 = _ValueHolder(0)
                    crop_x2 = _ValueHolder(-1)
                    crop_y2 = _ValueHolder(-1)

                    def on_crop_changed(x1: int, y1: int, x2: int, y2: int) -> None:
                        crop_x1.value = x1
                        crop_y1.value = y1
                        crop_x2.value = x2
                        crop_y2.value = y2

                    _crop_cb_ref[0] = on_crop_changed

                    # ── BNL input dict ─────────────────────────────────────────
                    fl_inputs: dict = {}
                    fl_valid: list  = []

                    for key, widget in [
                        ("fn_root",                  fn_root_el),
                        ("fn_data",                  fn_data_el),
                        ("theta_ls_dataset",         _ValueHolder("rot_angles")),
                        ("probe_energy",             probe_energy_el),
                        ("pixel_size_nm",            pixel_size_nm_el),
                        ("element_symbols",          elem_selector),
                        ("element_channel_indices",  _IndicesProxy(elem_selector)),
                        ("xrf_shell",                _ShellProxy(elem_selector)),
                        ("density",                  _DensityProxy(elem_selector)),
                        ("emission_energy",          _EmissionProxy(elem_selector)),
                        ("ic_channel_idx",           ic_selector),
                        ("crop_x_start",             crop_x1),
                        ("crop_x_end",               crop_x2),
                        ("crop_y_start",             crop_y1),
                        ("crop_y_end",               crop_y2),
                    ]:
                        fl_inputs[key] = widget
                        fl_valid.append(key)

                    # ── BNL FL Correction — method-specific params ────────────
                    ui.separator().classes("my-1")
                    ui.label("Processing").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Binning Factor", value=4, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("Spatial binning factor applied before attenuation calculation")
                        fl_inputs["binning_factor"] = el
                        fl_valid.append("binning_factor")

                        el = ui.number(
                            "Unit Scale", value=1e15, step=1e14, min=1, format="%.2e"
                        ).classes("flex-1")
                        el.tooltip("Scaling factor applied to projections (molar → femto-molar = 1e15)")
                        fl_inputs["scale"] = el
                        fl_valid.append("scale")

                    ui.separator().classes("my-1")
                    ui.label("Detector Mask Geometry").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    ui.label(
                        "If mask3D_N.h5 already exists in the working directory, "
                        "it will be loaded directly (skipping generation)."
                    ).classes("text-xs text-gray-500 italic")
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Horizontal Angle (°)", value=20.6, step=0.1,
                            min=0.1, format="%.1f"
                        ).classes("flex-1")
                        el.tooltip("Horizontal half-angle of the detector solid angle (degrees)")
                        fl_inputs["det_alfa"] = el
                        fl_valid.append("det_alfa")

                        el = ui.number(
                            "Vertical Angle (°)", value=20.6, step=0.1,
                            min=0.1, format="%.1f"
                        ).classes("flex-1")
                        el.tooltip("Vertical half-angle of the detector solid angle (degrees)")
                        fl_inputs["det_theta"] = el
                        fl_valid.append("det_theta")

                        el = ui.number(
                            "Mask Length (px)", value=200, step=10, min=10
                        ).classes("flex-1")
                        el.tooltip("Maximum radial length for the 3D detector mask")
                        fl_inputs["mask_length_maximum"] = el
                        fl_valid.append("mask_length_maximum")

                    ui.separator().classes("my-1")
                    ui.label("Initial Reconstruction (ASTRA)").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.select(
                            label="Method",
                            options=["EM_CUDA", "FBP_CUDA", "SIRT_CUDA", "FBP"],
                            value="EM_CUDA",
                        ).classes("flex-1")
                        el.tooltip("ASTRA tomography reconstruction algorithm")
                        fl_inputs["recon_method"] = el
                        fl_valid.append("recon_method")

                        el = ui.number(
                            "Iterations", value=16, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("MLEM/SIRT iterations for the initial reconstruction")
                        fl_inputs["recon_n_iter"] = el
                        fl_valid.append("recon_n_iter")

                    ui.separator().classes("my-1")
                    ui.label("Iterative Correction").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Correction Iters", value=4, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("Number of FL self-absorption correction iterations")
                        fl_inputs["n_correction_iters"] = el
                        fl_valid.append("n_correction_iters")

                        el = ui.number(
                            "MLEM Iters / Step", value=16, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("MLEM iterations per element per correction step")
                        fl_inputs["correction_n_iter"] = el
                        fl_valid.append("correction_n_iter")

                        el = ui.number(
                            "Border Pixels", value=5, step=1, min=0
                        ).classes("flex-1")
                        el.tooltip("Zero out this many pixels at each edge")
                        fl_inputs["border_pixels"] = el
                        fl_valid.append("border_pixels")

                        el = ui.number(
                            "Smooth Filter", value=3, step=2, min=1
                        ).classes("flex-1")
                        el.tooltip("Median filter kernel size applied between iterations")
                        fl_inputs["smooth_filter_size"] = el
                        fl_valid.append("smooth_filter_size")

                    # Always use GPU for BNL; hardcode defaults
                    fl_inputs["use_gpu"] = _ValueHolder(True)
                    fl_inputs["num_cpu"] = _ValueHolder(8)

            # ══════════════════════════════════════════════════════════════════
            # PANPAN TAB
            # ══════════════════════════════════════════════════════════════════
            with ui.tab_panel(panpan_tab):
                with ui.column().classes("w-full gap-2 p-2"):

                    recon_inputs: dict = {"fn_root": fn_root_el, "fn_data": fn_data_el, "probe_energy": probe_energy_el}
                    recon_valid: list  = ["fn_root", "fn_data", "probe_energy"]

                    recon_inputs["pixel_size_nm"] = pixel_size_nm_el

                    # Data Parameters ──────────────────────────────────────
                    ui.label("Data Parameters").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Probe Intensity", value=1.0e7, step=1e6, min=1
                        ).classes("flex-1")
                        el.tooltip("Incident probe intensity (photons/s)")
                        recon_inputs["probe_intensity"] = el
                        recon_valid.append("probe_intensity")

                        recon_xrt_sel.create_widget("XRT Ratio Channel").tooltip(
                            "HDF5 channel for XRT transmission ratio data"
                        )
                        recon_inputs["XRT_ratio_dataset_idx"] = recon_xrt_sel

                        recon_us_ic_sel.create_widget("Upstream IC Channel").tooltip(
                            "HDF5 channel for upstream ion chamber counts"
                        )
                        recon_inputs["scaler_counts_us_ic_dataset_idx"] = recon_us_ic_sel

                        recon_ds_ic_sel.create_widget("Downstream IC Channel").tooltip(
                            "HDF5 channel for downstream ion chamber counts"
                        )
                        recon_inputs["scaler_counts_ds_ic_dataset_idx"] = recon_ds_ic_sel

                    recon_inputs["theta_ls_dataset"] = _ValueHolder("rot_angles")
                    recon_inputs["channel_names"]   = _ValueHolder("elements")
                    recon_inputs["element_symbols"] = elem_selector
                    recon_inputs["emission_energy"] = _EmissionProxy(elem_selector)

                    ui.separator().classes("my-1")

                    # Detector Geometry ───────────────────────────────────
                    ui.label("Detector Geometry").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4 items-start"):
                        with ui.column().classes("flex-1 gap-2"):
                            with ui.row().classes("w-full gap-4"):
                                el = ui.number(
                                    "Detector Diameter (cm)", value=0.9, step=0.1, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Fluorescence detector diameter in cm")
                                recon_inputs["det_dia_cm"] = el
                                recon_valid.append("det_dia_cm")

                                el = ui.number(
                                    "Sample-Det Distance (cm)", value=1.6, step=0.1, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Distance from sample to detector in cm")
                                recon_inputs["det_from_sample_cm"] = el
                                recon_valid.append("det_from_sample_cm")

                                el = ui.number(
                                    "Det Pixel Spacing (cm)", value=0.4, step=0.05, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Spacing between detector pixels in cm")
                                recon_inputs["det_ds_spacing_cm"] = el
                                recon_valid.append("det_ds_spacing_cm")

                            el = ui.select(
                                label="Detector Side",
                                options={"positive": "Right", "negative": "Left"},
                                value="positive",
                            ).classes("w-full")
                            el.tooltip("Which side of the sample the detector is on")
                            recon_inputs["det_on_which_side"] = el
                            recon_valid.append("det_on_which_side")

                            el = ui.switch("Manual Detector Area", value=False)
                            el.tooltip("Skip solid-angle correction (data already calibrated)")
                            recon_inputs["manual_det_area"] = el
                            recon_valid.append("manual_det_area")

                        # 3D interactive diagram
                        with ui.column().classes("gap-0"):
                            recon_det_update = create_detector_diagram()

                    # Wire live diagram updates
                    _det_diagram_refs.append((recon_det_update, recon_inputs))

                    def _recon_det_refresh(_=None):
                        recon_det_update(
                            det_dia_cm=recon_inputs["det_dia_cm"].value or 0.9,
                            det_from_sample_cm=recon_inputs["det_from_sample_cm"].value or 1.6,
                            det_ds_spacing_cm=recon_inputs["det_ds_spacing_cm"].value or 0.4,
                            det_on_which_side=recon_inputs["det_on_which_side"].value or "positive",
                            sample_size_cm=_compute_sample_size_cm(),
                        )

                    recon_inputs["det_dia_cm"].on_value_change(_recon_det_refresh)
                    recon_inputs["det_from_sample_cm"].on_value_change(_recon_det_refresh)
                    recon_inputs["det_ds_spacing_cm"].on_value_change(_recon_det_refresh)
                    recon_inputs["det_on_which_side"].on_value_change(_recon_det_refresh)

                    ui.separator().classes("my-1")

                    # Reconstruction Parameters (Adam) ─────────────────────
                    ui.label("Reconstruction Parameters").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Epochs", value=300, step=10, min=1
                        ).classes("flex-1")
                        el.tooltip("Total number of reconstruction epochs")
                        recon_inputs["n_epochs"] = el
                        recon_valid.append("n_epochs")

                        el = ui.number(
                            "Save Every N", value=10, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("Save a checkpoint every N epochs")
                        recon_inputs["save_every_n_epochs"] = el
                        recon_valid.append("save_every_n_epochs")

                        el = ui.number(
                            "Minibatch Size", value=64, min=1
                        ).classes("flex-1")
                        el.tooltip(
                            "Number of voxel strips (from flattened H\u00d7W grid) per gradient update. "
                            "Total strips = height \u00d7 width. Smaller = more updates per epoch (SGD); "
                            "larger = fewer updates (full batch). Per-epoch time is similar either way."
                        )
                        recon_inputs["minibatch_size"] = el
                        recon_valid.append("minibatch_size")

                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Learning Rate", value=1.0e-3, step=1e-4,
                            min=1e-8, format="%.6f"
                        ).classes("flex-1")
                        el.tooltip("Adam optimizer learning rate")
                        recon_inputs["lr"] = el
                        recon_valid.append("lr")

                        el = ui.number(
                            "XRT Loss Weight (b\u2081)", value=0.0, step=100,
                            min=0.0, format="%.2e"
                        ).classes("flex-1")
                        el.tooltip(
                            "Weight of XRT transmission loss in combined objective: "
                            "loss = XRF + b\u2081\u00b7XRT. Use 0 to disable XRT loss; "
                            "typical experimental value: 1e4"
                        )
                        recon_inputs["b1"] = el
                        recon_valid.append("b1")

                        el = ui.number(
                            "XRT Data Scale (b\u2082)", value=1.0, step=0.1,
                            min=0.0, format="%.2e"
                        ).classes("flex-1")
                        el.tooltip(
                            "Scale factor applied to measured XRT transmission data "
                            "before computing loss"
                        )
                        recon_inputs["b2"] = el
                        recon_valid.append("b2")

                    with ui.row().classes("w-full gap-4"):
                        el = ui.select(
                            label="Initialization Kind",
                            options=["const", "rand", "randn"],
                            value="const",
                        ).classes("flex-1")
                        el.tooltip(
                            "How to initialize the concentration grid: "
                            "const = uniform value, rand = uniform random, randn = Gaussian random"
                        )
                        recon_inputs["ini_kind"] = el
                        recon_valid.append("ini_kind")

                        el = ui.number(
                            "Initial Constant", value=0.0,
                            step=0.01, format="%.4f"
                        ).classes("flex-1")
                        el.tooltip("Base concentration value for initialization")
                        recon_inputs["init_const"] = el
                        recon_valid.append("init_const")

                        el = ui.number(
                            "Random Amplitude", value=0.1,
                            step=0.01, min=0.0, format="%.4f"
                        ).classes("flex-1")
                        el.tooltip("Amplitude of random noise added to initial constant (used when ini_kind='rand' or 'randn')")
                        recon_inputs["ini_rand_amp"] = el
                        recon_valid.append("ini_rand_amp")

                    with ui.row().classes("w-full gap-8"):
                        el = ui.switch("Self-Absorption Correction", value=True)
                        el.tooltip("Account for self-absorption of fluorescent X-rays")
                        recon_inputs["selfAb"] = el
                        recon_valid.append("selfAb")

                        el = ui.switch("Probe Attenuation Correction", value=True)
                        el.tooltip("Account for probe beam attenuation through the sample")
                        recon_inputs["probe_att"] = el
                        recon_valid.append("probe_att")

                        el = ui.switch("Continue from Checkpoint", value=False)
                        el.tooltip("Resume from a previously saved checkpoint")
                        recon_inputs["cont_from_check_point"] = el
                        recon_valid.append("cont_from_check_point")

                        el = ui.switch("Use Saved Initial Guess", value=False)
                        el.tooltip("Load a previously saved initial concentration grid")
                        recon_inputs["use_saved_initial_guess"] = el
                        recon_valid.append("use_saved_initial_guess")

            # ══════════════════════════════════════════════════════════════════
            # DI TAB
            # ══════════════════════════════════════════════════════════════════
            with ui.tab_panel(di_tab):
                with ui.column().classes("w-full gap-2 p-2"):

                    di_inputs: dict = {"fn_root": fn_root_el, "fn_data": fn_data_el, "probe_energy": probe_energy_el}
                    di_valid: list  = ["fn_root", "fn_data", "probe_energy"]

                    di_inputs["pixel_size_nm"] = pixel_size_nm_el

                    # Data Parameters ──────────────────────────────────────
                    ui.label("Data Parameters").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Probe Intensity", value=1.0e7, step=1e6, min=1
                        ).classes("flex-1")
                        el.tooltip("Incident probe intensity (photons/s)")
                        di_inputs["probe_intensity"] = el
                        di_valid.append("probe_intensity")

                        di_xrt_sel.create_widget("XRT Ratio Channel").tooltip(
                            "Channel used as XRT transmission ratio (typically abs_ic)"
                        )
                        di_inputs["XRT_ratio_dataset_idx"] = di_xrt_sel

                        di_us_ic_sel.create_widget("Upstream IC Channel").tooltip(
                            "Channel for upstream ion chamber counts (typically us_ic)"
                        )
                        di_inputs["scaler_counts_us_ic_dataset_idx"] = di_us_ic_sel

                        di_ds_ic_sel.create_widget("Downstream IC Channel").tooltip(
                            "Channel for downstream ion chamber counts (typically ds_ic)"
                        )
                        di_inputs["scaler_counts_ds_ic_dataset_idx"] = di_ds_ic_sel

                    di_inputs["theta_ls_dataset"] = _ValueHolder("rot_angles")
                    di_inputs["channel_names"]   = _ValueHolder("elements")
                    di_inputs["element_symbols"] = elem_selector
                    di_inputs["emission_energy"] = _EmissionProxy(elem_selector)

                    ui.separator().classes("my-1")

                    # Detector Geometry ───────────────────────────────────
                    ui.label("Detector Geometry").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4 items-start"):
                        with ui.column().classes("flex-1 gap-2"):
                            with ui.row().classes("w-full gap-4"):
                                el = ui.number(
                                    "Detector Diameter (cm)", value=0.9, step=0.1, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Fluorescence detector diameter in cm")
                                di_inputs["det_dia_cm"] = el
                                di_valid.append("det_dia_cm")

                                el = ui.number(
                                    "Sample-Det Distance (cm)", value=1.6, step=0.1, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Distance from sample to detector in cm")
                                di_inputs["det_from_sample_cm"] = el
                                di_valid.append("det_from_sample_cm")

                                el = ui.number(
                                    "Det Pixel Spacing (cm)", value=0.4, step=0.05, min=0.01
                                ).classes("flex-1")
                                el.tooltip("Spacing between detector pixels in cm")
                                di_inputs["det_ds_spacing_cm"] = el
                                di_valid.append("det_ds_spacing_cm")

                            el = ui.select(
                                label="Detector Side",
                                options={"positive": "Right", "negative": "Left"},
                                value="positive",
                            ).classes("w-full")
                            el.tooltip("Which side of the sample the detector is on")
                            di_inputs["det_on_which_side"] = el
                            di_valid.append("det_on_which_side")

                            el = ui.switch("Manual Detector Area", value=False)
                            el.tooltip("Skip solid-angle correction (data already calibrated)")
                            di_inputs["manual_det_area"] = el
                            di_valid.append("manual_det_area")

                        # 3D interactive diagram
                        with ui.column().classes("gap-0"):
                            di_det_update = create_detector_diagram()

                    # Wire live diagram updates
                    _det_diagram_refs.append((di_det_update, di_inputs))

                    def _di_det_refresh(_=None):
                        di_det_update(
                            det_dia_cm=di_inputs["det_dia_cm"].value or 0.9,
                            det_from_sample_cm=di_inputs["det_from_sample_cm"].value or 1.6,
                            det_ds_spacing_cm=di_inputs["det_ds_spacing_cm"].value or 0.4,
                            det_on_which_side=di_inputs["det_on_which_side"].value or "positive",
                            sample_size_cm=_compute_sample_size_cm(),
                        )

                    di_inputs["det_dia_cm"].on_value_change(_di_det_refresh)
                    di_inputs["det_from_sample_cm"].on_value_change(_di_det_refresh)
                    di_inputs["det_ds_spacing_cm"].on_value_change(_di_det_refresh)
                    di_inputs["det_on_which_side"].on_value_change(_di_det_refresh)

                    ui.separator().classes("my-1")

                    # Reconstruction Parameters (L-BFGS bi-level) ──────────
                    ui.label("Reconstruction Parameters").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Outer Epochs", value=5, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip(
                            "Number of outer bi-level iterations — each outer epoch "
                            "freezes self-absorption and runs L-BFGS inner minimization"
                        )
                        di_inputs["n_outer_epochs"] = el
                        di_valid.append("n_outer_epochs")

                        el = ui.number(
                            "L-BFGS Max Inner Iters", value=20, step=5, min=1
                        ).classes("flex-1")
                        el.tooltip("Maximum L-BFGS inner iterations per outer epoch")
                        di_inputs["lbfgs_n_iter"] = el
                        di_valid.append("lbfgs_n_iter")

                        el = ui.number(
                            "L-BFGS History Size", value=10, step=5, min=1
                        ).classes("flex-1")
                        el.tooltip("Past gradients stored for Hessian approximation")
                        di_inputs["lbfgs_history"] = el
                        di_valid.append("lbfgs_history")

                    with ui.row().classes("w-full gap-4"):
                        el = ui.select(
                            label="Loss Type",
                            options=["poisson", "ls"],
                            value="poisson",
                        ).classes("flex-1")
                        el.tooltip(
                            "XRF loss function: 'poisson' = Poisson NLL, "
                            "'ls' = least squares MSE"
                        )
                        di_inputs["loss_type"] = el
                        di_valid.append("loss_type")

                        el = ui.number(
                            "XRT Weight (β₁)", value=1.0, step=0.1,
                            min=0.0, format="%.3f"
                        ).classes("flex-1")
                        el.tooltip("Weight for the XRT transmission loss term")
                        di_inputs["beta1_xrt"] = el
                        di_valid.append("beta1_xrt")

                        el = ui.number(
                            "Tikhonov λ", value=0.0, step=0.001,
                            min=0.0, format="%.5f"
                        ).classes("flex-1")
                        el.tooltip("L2 regularisation weight (0 to disable)")
                        di_inputs["tikhonov_lambda"] = el
                        di_valid.append("tikhonov_lambda")

                    with ui.row().classes("w-full gap-4"):
                        el = ui.number(
                            "Save Every N Epochs", value=1, step=1, min=1
                        ).classes("flex-1")
                        el.tooltip("Save checkpoint after every N outer epochs")
                        di_inputs["save_every_n_epochs"] = el
                        di_valid.append("save_every_n_epochs")

                        el = ui.number(
                            "Minibatch Size", value=64, min=1
                        ).classes("flex-1")
                        el.tooltip(
                            "Number of voxel strips per gradient update (same as Panpan). "
                            "Controls how many strips of the H\u00d7W grid are processed per L-BFGS step."
                        )
                        di_inputs["minibatch_size"] = el
                        di_valid.append("minibatch_size")

                    with ui.row().classes("w-full gap-4"):
                        el = ui.select(
                            label="Initialization Kind",
                            options=["const", "rand"],
                            value="const",
                        ).classes("flex-1")
                        el.tooltip("How to initialize the concentration grid")
                        di_inputs["ini_kind"] = el
                        di_valid.append("ini_kind")

                        el = ui.number(
                            "Initial Constant", value=0.0,
                            step=0.01, format="%.4f"
                        ).classes("flex-1")
                        el.tooltip("Initial concentration value when ini_kind='const'")
                        di_inputs["init_const"] = el
                        di_valid.append("init_const")

                    with ui.row().classes("w-full gap-8"):
                        el = ui.switch("Self-Absorption Correction", value=True)
                        el.tooltip("Account for self-absorption of fluorescent X-rays")
                        di_inputs["selfAb"] = el
                        di_valid.append("selfAb")

                        el = ui.switch("Probe Attenuation Correction", value=True)
                        el.tooltip("Account for probe beam attenuation through the sample")
                        di_inputs["probe_att"] = el
                        di_valid.append("probe_att")

                        el = ui.switch("Continue from Checkpoint", value=False)
                        el.tooltip("Resume from a previously saved checkpoint")
                        di_inputs["cont_from_check_point"] = el
                        di_valid.append("cont_from_check_point")

                        el = ui.switch("Use Saved Initial Guess", value=False)
                        el.tooltip("Load a previously saved initial concentration grid")
                        di_inputs["use_saved_initial_guess"] = el
                        di_valid.append("use_saved_initial_guess")

        # ══════════════════════════════════════════════════════════════════════
        # SHARED RUN/STOP + PROGRESS (outside tabs, above status log)
        # Single button pair dispatches to the active tab's method.
        # ══════════════════════════════════════════════════════════════════════

        # ── Combined: GPU controls + Session bar (one card, ptycho style) ──────
        auto_gpu = {"enabled": False}
        selected_session = {"id": ""}
        _gpu_panel_open = {"value": False}

        combined_card = ui.card().classes("w-full")
        with combined_card:

            # ── GPU controls row ──────────────────────────────────────────────
            with ui.row().classes("w-full gap-3 items-center flex-wrap"):
                with ui.row().classes("items-center gap-1"):
                    check_gpu_btn = ui.button(icon="memory").props(
                        "flat dense round color=primary"
                    ).tooltip("Toggle GPU status panel")
                    ui.label("GPU").classes("text-sm font-bold whitespace-nowrap")
                    gpu_id_input = ui.number(value=0, step=1, min=-1).classes("w-20").props("outlined dense")
                    gpu_id_input.tooltip("GPU device ID (-1 = CPU)")

                    def _toggle_auto_gpu(e):
                        auto_gpu["enabled"] = e.value
                        gpu_id_input.set_enabled(not e.value)
                    ui.switch("Auto").classes("ml-1").props("dense").on_value_change(
                        _toggle_auto_gpu
                    ).tooltip("Auto-select GPU with lowest memory usage")

                with ui.row().classes("items-center gap-1"):
                    ui.label("Sessions").classes("text-sm font-bold whitespace-nowrap")
                    n_sessions_input = ui.number(value=1, step=1, min=1, max=8).classes("w-20").props("outlined dense")
                    n_sessions_input.tooltip("Number of parallel sessions to launch")

                shared_run_btn = ui.button(
                    "Start Reconstruction", icon="play_arrow", color="green"
                ).props("unelevated no-caps size=lg dense").classes("flex-1")
                shared_stop_btn = ui.button(
                    "Stop Reconstruction", icon="stop", color="red"
                ).props("unelevated no-caps size=lg dense").classes("flex-1")

            # ── GPU status panel (hidden column, toggled by memory icon) ──────
            gpu_status_outer = ui.column().classes("w-full gap-0 px-1 mt-2")
            gpu_status_outer.visible = False

            with gpu_status_outer:
                with ui.row().classes("w-full items-center justify-between pb-1"):
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("memory", size="xs").classes("text-gray-400")
                        ui.label("GPU Status").classes(
                            "text-xs font-semibold text-gray-500 uppercase tracking-wide"
                        )
                    ui.button(
                        icon="refresh",
                        on_click=lambda: asyncio.ensure_future(_refresh_gpu_status(show_spinner=True)),
                    ).props("flat dense round size=xs color=primary").tooltip("Refresh GPU status")
                gpu_rows_container = ui.column().classes("w-full gap-0")

            async def _refresh_gpu_status(show_spinner=False):
                gpu_rows_container.clear()
                try:
                    data = await api.gpu_status()
                    gpus = data.get("gpus", [])
                    if not gpus:
                        with gpu_rows_container:
                            ui.label(data.get("error", "No GPUs found")).classes("text-xs text-red-500")
                        return
                    with gpu_rows_container:
                        for gpu in gpus:
                            idx = gpu["index"]
                            name = gpu.get("name", "Unknown GPU")
                            mem_used = gpu.get("memory_used_mb", 0)
                            mem_total = gpu.get("memory_total_mb", 1)
                            util = gpu.get("utilization_pct", -1)
                            gpu_error = gpu.get("error")
                            mem_pct = mem_used / mem_total if mem_total > 0 else 0
                            with ui.row().classes("w-full items-center gap-3 py-1").style(
                                "border-bottom: 1px solid #f0f0f0;"
                            ):
                                ui.chip(f"GPU {idx}", color="blue-grey").props("dense outline").style(
                                    "font-size: 11px; padding: 2px 8px; line-height: 1.6;"
                                )
                                ui.label(name).classes("text-sm font-medium").style("min-width: 160px;")
                                num_proc = gpu.get("num_processes", 0)
                                proc_str = f"{num_proc} process" if num_proc == 1 else f"{num_proc} processes"
                                ui.label(proc_str).classes("text-xs text-gray-400 whitespace-nowrap").style(
                                    "min-width: 70px;"
                                )
                                with ui.row().classes("flex-1 items-center gap-2"):
                                    mem_color = "primary" if mem_pct < 0.8 else "negative"
                                    ui.linear_progress(
                                        value=mem_pct, color=mem_color, show_value=False
                                    ).props("size=8px rounded").style("flex: 1;")
                                    ui.label(f"{mem_used:,} / {mem_total:,} MB").classes(
                                        "text-xs text-gray-400 whitespace-nowrap"
                                    )
                                if util >= 0:
                                    util_color = (
                                        "positive" if util < 50 else ("warning" if util < 80 else "negative")
                                    )
                                    ui.chip(f"{util}% util", color=util_color).props("dense")
                                else:
                                    ui.chip("N/A", color="grey").props("dense").tooltip(
                                        "Utilization unavailable"
                                    )
                                if gpu_error:
                                    ui.icon("warning", color="negative", size="xs").tooltip(gpu_error)
                except Exception as exc:
                    with gpu_rows_container:
                        ui.label(f"Error: {exc}").classes("text-xs text-red-500")

            async def _toggle_gpu_panel():
                if _gpu_panel_open["value"]:
                    _gpu_panel_open["value"] = False
                    gpu_status_outer.visible = False
                else:
                    _gpu_panel_open["value"] = True
                    gpu_status_outer.visible = True
                    await _refresh_gpu_status(show_spinner=True)

            check_gpu_btn.on_click(lambda: asyncio.ensure_future(_toggle_gpu_panel()))

            async def _auto_refresh_gpu():
                if _gpu_panel_open["value"]:
                    await _refresh_gpu_status(show_spinner=False)

            ui.timer(5.0, _auto_refresh_gpu)

            # ── Session bar (inside same card) ────────────────────────────────
            ui.separator().classes("my-2")

            def _on_session_select(sid: str):
                selected_session["id"] = sid
                update_session_bar(_session_bar_data["sessions"], sid)

            async def _on_session_remove(sid: str):
                try:
                    await api.remove_session(sid)
                except Exception:
                    pass
                if selected_session["id"] == sid:
                    selected_session["id"] = ""
                try:
                    sessions = await api.list_sessions()
                    _session_bar_data["sessions"] = sessions
                    update_session_bar(sessions, selected_session["id"])
                except Exception:
                    pass

            async def _on_clear_finished():
                try:
                    await api.clear_finished_sessions()
                except Exception:
                    pass
                try:
                    sessions = await api.list_sessions()
                    _session_bar_data["sessions"] = sessions
                    update_session_bar(sessions, selected_session["id"])
                except Exception:
                    pass

            _session_bar_data = {"sessions": []}
            _session_bar_container, update_session_bar = create_session_bar(
                on_select=_on_session_select,
                on_remove=lambda sid: asyncio.ensure_future(_on_session_remove(sid)),
                on_clear_finished=lambda: asyncio.ensure_future(_on_clear_finished()),
            )

        with combined_card:
            ui.separator().classes("my-2")
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("speed", size="xs").classes("text-gray-400")
                ui.label("Progress").classes("section-header").style("margin-bottom: 0;")
                ui.space()
                shared_method_label = ui.label("BNL").classes("text-xs text-gray-400")
            shared_step_info = ui.label("No reconstruction running").classes(
                "text-gray-500 text-lg font-semibold w-full mb-1"
            )
            shared_progress_bar = ui.linear_progress(value=0, show_value=False).classes("mb-1")
            shared_progress_label = ui.label("0%").classes("text-center text-sm")

        def update_shared_progress():
            active = tabs.value
            if active == "BNL":
                pct = fl_state.progress_percent
                shared_method_label.set_text("BNL")
                if fl_state.is_running or fl_state.current_step > 0:
                    step = (
                        fl_state.step_label
                        or f"Step {fl_state.current_step}/{fl_state.total_steps}"
                    )
                    shared_step_info.set_text(f"{step}  ({pct:.0f}%)")
                else:
                    shared_step_info.set_text("No correction running")
            elif active == "Panpan":
                pct = recon_state.progress_percent
                shared_method_label.set_text("Panpan")
                if recon_state.is_running or recon_state.current_epoch > 0:
                    shared_step_info.set_text(
                        f"Epoch {recon_state.current_epoch}/{recon_state.total_epochs}  ({pct:.0f}%)"
                    )
                else:
                    shared_step_info.set_text("No reconstruction running")
            elif active == "Wendy":
                pct = di_state.progress_percent
                shared_method_label.set_text("Di et al. 2017")
                if di_state.is_running or di_state.current_epoch > 0:
                    shared_step_info.set_text(
                        f"Epoch {di_state.current_epoch}/{di_state.total_epochs}  ({pct:.0f}%)"
                    )
                else:
                    shared_step_info.set_text("No Di reconstruction running")
            else:
                pct = 0.0
            shared_progress_bar.set_value(pct / 100.0)
            shared_progress_label.set_text(f"{pct:.1f}%")

        def update_shared_button():
            active = tabs.value
            if active == "BNL":
                st = fl_state
            elif active == "Panpan":
                st = recon_state
            else:
                st = di_state
            if st.is_running:
                shared_run_btn.props("color=orange")
                shared_run_btn.disable()
            elif st.button_status == "processing":
                if time.time() - st.button_timestamp > 10:
                    shared_run_btn.props("color=green")
                    shared_run_btn.enable()
                    st.button_status = "idle"
            else:
                shared_run_btn.props("color=green")
                shared_run_btn.enable()

        async def _resolve_gpu_id() -> int:
            """Return the GPU ID to use, auto-detecting if Auto is enabled."""
            if auto_gpu["enabled"]:
                try:
                    data = await api.gpu_status()
                    gpus = data.get("gpus", [])
                    if gpus:
                        best = min(gpus, key=lambda g: g.get("memory_used_mb", 0))
                        return best["index"]
                except Exception:
                    pass
            return int(gpu_id_input.value or 0)

        async def _on_shared_run():
            active = tabs.value
            n_sessions = max(1, int(n_sessions_input.value or 1))
            shared_run_btn.props("color=orange")
            shared_run_btn.disable()

            if active == "BNL":
                fl_state.button_status = "processing"
                fl_state.button_timestamp = time.time()
                for i in range(n_sessions):
                    if i > 0:
                        await asyncio.sleep(20)
                    params = _collect_fl_params(fl_inputs, fl_valid)
                    params["gpu_id"] = await _resolve_gpu_id()
                    try:
                        fresh = []
                        fresh = append_to_message_list(fresh, f"Starting FL correction (session {i+1}/{n_sessions})...", level="INFO")
                        fresh = append_to_message_list(fresh, f"  fn_root: {params.get('fn_root', '')}", level="INFO")
                        fresh = append_to_message_list(fresh, f"  n_correction_iters: {params.get('n_correction_iters', 4)}", level="INFO")
                        fresh = append_to_message_list(fresh, f"  gpu_id: {params['gpu_id']}", level="INFO")
                        await api.setup_fl_correction(params)
                        run_resp = await api.run_fl_correction()
                        status_msg = run_resp.get("status", "FL correction started.")
                        sid = run_resp.get("session_id", "")
                        fresh = append_to_message_list(fresh, status_msg, level="WORKER")
                        fl_state.messages = fresh
                        fl_state.results_ready = False
                        fl_state.recon_file = ""
                        fl_state.current_step = 0
                        fl_state.total_steps = 0
                        fl_state.step_label = ""
                        if sid:
                            selected_session["id"] = sid
                    except Exception as e:
                        fl_state.messages = append_to_message_list(
                            fl_state.messages, f"Error starting FL correction: {e}", level="ERROR"
                        )
                        shared_run_btn.props("color=green")
                        shared_run_btn.enable()
                        fl_state.button_status = "idle"
                        break

            elif active == "Panpan":
                recon_state.button_status = "processing"
                recon_state.button_timestamp = time.time()
                for i in range(n_sessions):
                    if i > 0:
                        await asyncio.sleep(20)
                    params = _collect_recon_params(recon_inputs, recon_valid)
                    params["gpu_id"] = await _resolve_gpu_id()
                    try:
                        fresh = []
                        fresh = append_to_message_list(fresh, f"Starting reconstruction (session {i+1}/{n_sessions})...", level="INFO")
                        fresh = append_to_message_list(fresh, f"  gpu_id: {params['gpu_id']}", level="INFO")
                        await api.setup_reconstruction(params)
                        run_resp = await api.run_reconstruction()
                        status_msg = run_resp.get("status", "Reconstruction started successfully.")
                        sid = run_resp.get("session_id", "")
                        fresh = append_to_message_list(fresh, status_msg, level="WORKER")
                        recon_state.messages = fresh
                        recon_state.results_ready = False
                        if sid:
                            selected_session["id"] = sid
                    except Exception as e:
                        recon_state.messages = append_to_message_list(
                            recon_state.messages, f"Error starting reconstruction: {e}", level="ERROR"
                        )
                        shared_run_btn.props("color=green")
                        shared_run_btn.enable()
                        recon_state.button_status = "idle"
                        break

            elif active == "Wendy":
                di_state.button_status = "processing"
                di_state.button_timestamp = time.time()
                for i in range(n_sessions):
                    if i > 0:
                        await asyncio.sleep(20)
                    params = _collect_di_params(di_inputs, di_valid)
                    params["gpu_id"] = await _resolve_gpu_id()
                    try:
                        fresh = []
                        fresh = append_to_message_list(fresh, f"Starting Di reconstruction (session {i+1}/{n_sessions})...", level="INFO")
                        fresh = append_to_message_list(fresh, f"  gpu_id: {params['gpu_id']}", level="INFO")
                        await api.setup_di_reconstruction(params)
                        run_resp = await api.run_di_reconstruction()
                        status_msg = run_resp.get("status", "Di reconstruction started successfully.")
                        sid = run_resp.get("session_id", "")
                        fresh = append_to_message_list(fresh, status_msg, level="WORKER")
                        di_state.messages = fresh
                        di_state.results_ready = False
                        if sid:
                            selected_session["id"] = sid
                    except Exception as e:
                        di_state.messages = append_to_message_list(
                            di_state.messages, f"Error starting Di reconstruction: {e}", level="ERROR"
                        )
                        shared_run_btn.props("color=green")
                        shared_run_btn.enable()
                        di_state.button_status = "idle"
                        break

        shared_run_btn.on_click(_on_shared_run)

        async def _on_shared_stop():
            sid = selected_session["id"]
            active = tabs.value

            # Choose target state for status messages
            if active == "BNL":
                st = fl_state
            elif active == "Panpan":
                st = recon_state
            else:
                st = di_state

            if sid:
                # Stop the specifically selected session
                st.messages = append_to_message_list(
                    st.messages, f"Stopping session {sid[:8]}...", level="WARNING"
                )
                try:
                    resp = await api.stop_session(sid)
                    st.messages = append_to_message_list(
                        st.messages, resp.get("status", "Stopped."), level="SUCCESS"
                    )
                except Exception as e:
                    st.messages = append_to_message_list(
                        st.messages, f"Error stopping session: {e}", level="ERROR"
                    )
            else:
                # No session selected — fall back to method-based stop
                if active == "BNL":
                    fl_state.messages = append_to_message_list(
                        fl_state.messages, "Stopping FL correction...", level="WARNING"
                    )
                    try:
                        resp = await api.stop_fl_correction()
                        fl_state.messages = append_to_message_list(
                            fl_state.messages, resp.get("status", "Stopped."), level="SUCCESS"
                        )
                    except Exception as e:
                        fl_state.messages = append_to_message_list(
                            fl_state.messages, f"Error stopping: {e}", level="ERROR"
                        )
                elif active == "Panpan":
                    recon_state.messages = append_to_message_list(
                        recon_state.messages, "Stopping reconstruction...", level="WARNING"
                    )
                    try:
                        resp = await api.stop_reconstruction()
                        recon_state.messages = append_to_message_list(
                            recon_state.messages, resp.get("status", "Reconstruction stopped."), level="SUCCESS"
                        )
                    except Exception as e:
                        recon_state.messages = append_to_message_list(
                            recon_state.messages, f"Error stopping reconstruction: {e}", level="ERROR"
                        )
                elif active == "Wendy":
                    di_state.messages = append_to_message_list(
                        di_state.messages, "Stopping Di reconstruction...", level="WARNING"
                    )
                    try:
                        resp = await api.stop_di_reconstruction()
                        di_state.messages = append_to_message_list(
                            di_state.messages, resp.get("status", "Di reconstruction stopped."), level="SUCCESS"
                        )
                    except Exception as e:
                        di_state.messages = append_to_message_list(
                            di_state.messages, f"Error stopping Di reconstruction: {e}", level="ERROR"
                        )

            st.button_status = "idle"
            st.is_busy = False
            shared_run_btn.props("color=green")
            shared_run_btn.enable()

        shared_stop_btn.on_click(_on_shared_stop)

        # Update shared progress and button when tab changes
        def on_tab_change(e):
            _saved_tab["value"] = e.value
            update_shared_progress()
            update_shared_button()

        tabs.on_value_change(on_tab_change)

        with combined_card:
            ui.separator().classes("my-2")

            def _on_log_clear():
                fl_state.messages.clear()
                recon_state.messages.clear()
                di_state.messages.clear()
                fl_err["log_offset"] = 0
                recon_err["log_offset"] = 0
                di_err["log_offset"] = 0

            _log_area, update_log = create_status_log(log_state, on_clear=_on_log_clear)

        with combined_card:
            ui.separator().classes("my-2")
            _result_container, update_result_viewer = create_result_viewer()

    # ── Persist & restore state across page navigations ────────────────────────
    # (follows ptycho-gui-mic pattern: module-level dicts + restore on entry)

    def _save_param(key):
        """Return a callback that saves a parameter value to module-level dict."""
        def _handler(e):
            _saved_params[key] = e.value
        return _handler

    # Shared inputs
    for _key, _el in [("h5_path", h5_path_el), ("probe_energy", probe_energy_el),
                      ("pixel_size_nm", pixel_size_nm_el), ("host", host_input),
                      ("port", port_input), ("api_key", api_key_input),
                      ("gpu_id", gpu_id_input)]:
        if _key in _saved_params:
            _el.set_value(_saved_params[_key])
        _el.on_value_change(_save_param(_key))

    # Method-specific inputs (skip proxy objects — only real NiceGUI widgets)
    for _prefix, _inputs, _form_keys in [
        ("fl", fl_inputs, _FL_FORM_KEYS),
        ("recon", recon_inputs, _RECON_FORM_KEYS),
        ("di", di_inputs, _DI_FORM_KEYS),
    ]:
        for _key in _form_keys:
            _el = _inputs.get(_key)
            if _el is None:
                continue
            _skey = f"{_prefix}.{_key}"
            if _skey in _saved_params:
                _el.set_value(_saved_params[_skey])
            _el.on_value_change(_save_param(_skey))

    # Restore backend connection
    if "host" in _saved_params or "port" in _saved_params:
        h = _saved_params.get("host", init_host)
        p = _saved_params.get("port", init_port)
        api.set_endpoint(f"http://{h}:{p}")
        if "api_key" in _saved_params:
            api.set_api_key(_saved_params["api_key"])
        connection_label.set_text(f"{api.endpoint} (reconnecting...)")

    # Restore session & log state
    selected_session["id"] = _saved_session_state.get("selected_id", "")
    fl_state.messages = list(_saved_session_state.get("fl_messages", []))
    recon_state.messages = list(_saved_session_state.get("recon_messages", []))
    di_state.messages = list(_saved_session_state.get("di_messages", []))
    fl_err["log_offset"] = _saved_session_state.get("fl_log_offset", 0)
    recon_err["log_offset"] = _saved_session_state.get("recon_log_offset", 0)
    di_err["log_offset"] = _saved_session_state.get("di_log_offset", 0)
    fl_err["reported"] = _saved_session_state.get("fl_err_reported", False)
    recon_err["reported"] = _saved_session_state.get("recon_err_reported", False)
    di_err["reported"] = _saved_session_state.get("di_err_reported", False)

    # Restore active tab
    if _saved_tab["value"] != "BNL":
        tabs.set_value(_saved_tab["value"])

    # Render restored state immediately (don't wait for first poll)
    log_state.messages = fl_state.messages + recon_state.messages + di_state.messages
    update_log()
    update_shared_progress()
    update_shared_button()

    # Auto-reload H5 data if file path was restored (restores data preview,
    # element tiles, channel dropdown, angle slider — the full Data section)
    if "h5_path" in _saved_params and _saved_params["h5_path"]:
        async def _auto_reload_h5():
            await _h5_load_file()
        ui.timer(0.8, _auto_reload_h5, once=True)

    # ── Polling timer ─────────────────────────────────────────────────────────
    async def poll_backend():

        # ── BNL (also serves as the connectivity probe) ───────────────────────
        try:
            progress_data = await api.get_fl_progress()
            fl_state.is_running       = progress_data.get("is_running", False)
            fl_state.current_step     = progress_data.get("current_step", 0)
            fl_state.total_steps      = progress_data.get("total_steps", 0)
            fl_state.step_label       = progress_data.get("step_label", "")
            fl_state.progress_percent = progress_data.get("progress_percent", 0.0)

            try:
                fl_status = await api.get_fl_status()
                fl_error  = fl_status.get("error")
                if fl_error and not fl_err["reported"]:
                    fl_err["reported"] = True
                    fl_state.messages = append_to_message_list(
                        fl_state.messages, f"[BNL] {fl_error}", level="ERROR"
                    )
                    ui.notify(
                        "BNL FL correction failed. See status log.",
                        type="negative", position="top", timeout=8000,
                    )
                    fl_state.button_status = "idle"
                    fl_state.is_busy = False
                elif not fl_error:
                    fl_err["reported"] = False
            except Exception:
                pass

            try:
                results_data = await api.get_fl_results()
                if results_data.get("results_ready") and not fl_state.results_ready:
                    fl_state.recon_file  = results_data.get("recon_file", "")
                    fl_state.results_ready = True
                    ui.notify(
                        "BNL FL correction complete!",
                        type="positive", position="top", timeout=5000,
                    )
            except Exception:
                pass

            if fl_state.is_running:
                try:
                    worker_data = await api.get_fl_worker_status()
                    logs = worker_data.get("worker_logs", [])
                    if len(logs) < fl_err["log_offset"]:
                        fl_err["log_offset"] = 0
                    for entry in logs[fl_err["log_offset"]:]:
                        msg   = entry.get("message", str(entry)) if isinstance(entry, dict) else str(entry)
                        level = entry.get("level", "WORKER")       if isinstance(entry, dict) else "WORKER"
                        fl_state.messages = append_to_message_list(fl_state.messages, msg, level=level)
                    fl_err["log_offset"] = len(logs)
                except Exception:
                    pass

            # Mark backend connected
            if not backend_connected["value"]:
                backend_connected["value"] = True
                backend_connected["error_logged"] = False
                connection_label.set_text(f"{api.endpoint} (connected)")
                connection_label.classes(
                    remove="text-red-500 text-gray-400 text-orange-500",
                    add="text-green-600",
                )

        except httpx.HTTPStatusError as e:
            if not backend_connected["error_logged"]:
                backend_connected["error_logged"] = True
                backend_connected["value"] = False
                lbl = (
                    "(invalid API key)"
                    if e.response.status_code == 403
                    else f"(error {e.response.status_code})"
                )
                connection_label.set_text(f"{api.endpoint} {lbl}")
                connection_label.classes(
                    remove="text-green-600 text-gray-400",
                    add="text-orange-500" if e.response.status_code == 403 else "text-red-500",
                )
        except Exception:
            if not backend_connected["error_logged"]:
                backend_connected["error_logged"] = True
                backend_connected["value"] = False
                connection_label.set_text(f"{api.endpoint} (not reachable)")
                connection_label.classes(
                    remove="text-green-600 text-gray-400 text-orange-500",
                    add="text-red-500",
                )

        # ── Panpan ────────────────────────────────────────────────────────────
        try:
            progress_data = await api.get_recon_progress()
            recon_state.progress_percent = progress_data.get("progress_percent", 0)
            recon_state.is_running       = progress_data.get("is_running", False)
            recon_state.current_epoch    = progress_data.get("current_epoch", 0)
            recon_state.total_epochs     = progress_data.get("total_epochs", 0)

            try:
                recon_status = await api.get_recon_status()
                recon_error  = recon_status.get("error")
                if recon_error and not recon_err["reported"]:
                    recon_err["reported"] = True
                    recon_state.messages = append_to_message_list(
                        recon_state.messages, f"[Panpan] {recon_error}", level="ERROR"
                    )
                    ui.notify(
                        "Panpan reconstruction failed. See status log.",
                        type="negative", position="top", timeout=8000,
                    )
                    recon_state.button_status = "idle"
                    recon_state.is_busy = False
                elif not recon_error:
                    recon_err["reported"] = False
            except Exception:
                pass

            try:
                results_data = await api.get_recon_results()
                if results_data.get("results_ready") and not recon_state.results_ready:
                    recon_state.recon_file   = results_data.get("recon_file", "")
                    recon_state.results_ready = True
                    ui.notify(
                        "Panpan reconstruction complete!",
                        type="positive", position="top", timeout=5000,
                    )
            except Exception:
                pass

            if recon_state.is_running:
                try:
                    worker_data = await api.get_recon_worker_status()
                    logs = worker_data.get("worker_logs", [])
                    if len(logs) < recon_err["log_offset"]:
                        recon_err["log_offset"] = 0
                    for entry in logs[recon_err["log_offset"]:]:
                        msg   = entry.get("message", str(entry)) if isinstance(entry, dict) else str(entry)
                        level = entry.get("level", "WORKER")       if isinstance(entry, dict) else "WORKER"
                        recon_state.messages = append_to_message_list(recon_state.messages, msg, level=level)
                    recon_err["log_offset"] = len(logs)
                except Exception:
                    pass

        except Exception:
            pass

        # ── Di ────────────────────────────────────────────────────────────────
        try:
            progress_data = await api.get_di_recon_progress()
            di_state.is_running    = progress_data.get("is_running", False)
            di_state.current_epoch = progress_data.get("current_epoch", 0)
            di_state.total_epochs  = progress_data.get("total_epochs", 0)
            di_state.progress_percent = (
                di_state.current_epoch / di_state.total_epochs * 100.0
                if di_state.total_epochs > 0 else 0.0
            )

            try:
                di_status = await api.get_di_recon_status()
                di_error  = di_status.get("error")
                if di_error and not di_err["reported"]:
                    di_err["reported"] = True
                    di_state.messages = append_to_message_list(
                        di_state.messages, f"[Di] {di_error}", level="ERROR"
                    )
                    ui.notify(
                        "Di reconstruction failed. See status log.",
                        type="negative", position="top", timeout=8000,
                    )
                    di_state.button_status = "idle"
                    di_state.is_busy = False
                elif not di_error:
                    di_err["reported"] = False
            except Exception:
                pass

            try:
                results_data = await api.get_di_recon_results()
                recon_file   = results_data.get("recon_file", "")
                if recon_file and not results_data.get("error") and not di_state.results_ready:
                    di_state.recon_file   = recon_file
                    di_state.results_ready = True
                    ui.notify(
                        "Di reconstruction complete!",
                        type="positive", position="top", timeout=5000,
                    )
            except Exception:
                pass

            if di_state.is_running:
                try:
                    worker_data = await api.get_di_recon_worker_status()
                    logs = worker_data.get("logs", [])
                    if len(logs) < di_err["log_offset"]:
                        di_err["log_offset"] = 0
                    for entry in logs[di_err["log_offset"]:]:
                        msg   = entry.get("message", str(entry)) if isinstance(entry, dict) else str(entry)
                        level = entry.get("level", "WORKER")       if isinstance(entry, dict) else "WORKER"
                        di_state.messages = append_to_message_list(di_state.messages, msg, level=level)
                    di_err["log_offset"] = len(logs)
                except Exception:
                    pass

        except Exception:
            pass

        # ── Update UI ─────────────────────────────────────────────────────────
        # Merge all method messages into the shared status log
        log_state.messages = (
            fl_state.messages + recon_state.messages + di_state.messages
        )
        update_log()
        update_shared_progress()
        update_shared_button()

        # ── Update session bar ────────────────────────────────────────────────
        try:
            sessions = await api.list_sessions()
            _session_bar_data["sessions"] = sessions
            update_session_bar(sessions, selected_session["id"])
        except Exception:
            pass

        # ── Update result viewer ─────────────────────────────────────────────
        sid = selected_session["id"]
        if sid:
            await update_result_viewer(api, sid)

        # ── Save session state for navigation persistence ──────────────────────
        _saved_session_state["selected_id"] = selected_session["id"]
        _saved_session_state["fl_messages"] = list(fl_state.messages)
        _saved_session_state["recon_messages"] = list(recon_state.messages)
        _saved_session_state["di_messages"] = list(di_state.messages)
        _saved_session_state["fl_log_offset"] = fl_err["log_offset"]
        _saved_session_state["recon_log_offset"] = recon_err["log_offset"]
        _saved_session_state["di_log_offset"] = di_err["log_offset"]
        _saved_session_state["fl_err_reported"] = fl_err["reported"]
        _saved_session_state["recon_err_reported"] = recon_err["reported"]
        _saved_session_state["di_err_reported"] = di_err["reported"]

    ui.timer(3.0, poll_backend)
