"""FL self-absorption correction parameter form (BNL)."""

from nicegui import ui
from ..state import FLState
from .h5_inspector import create_h5_inspector

# ── Periodic-table colour palette ─────────────────────────────────────────────
_ELEM_COLORS: dict[str, str] = {}
for _s in "H Li Na K Rb Cs Fr".split():
    _ELEM_COLORS[_s] = "#f4a8a8"          # alkali metals
for _s in "Be Mg Ca Sr Ba Ra".split():
    _ELEM_COLORS[_s] = "#f4a8a8"          # alkaline-earth metals (same group)
for _s in (
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn"
).split():
    _ELEM_COLORS[_s] = "#aacce8"          # transition metals
for _s in "Al Ga In Sn Tl Pb Bi Po Nh Fl Mc Lv".split():
    _ELEM_COLORS[_s] = "#a8d8b0"          # post-transition metals
for _s in "B Si Ge As Se Sb Te At Ts".split():
    _ELEM_COLORS[_s] = "#f5e47a"          # metalloids
for _s in "C N O F P S Cl Br I".split():
    _ELEM_COLORS[_s] = "#d4eaa8"          # nonmetals
for _s in "He Ne Ar Kr Xe Rn Og".split():
    _ELEM_COLORS[_s] = "#d4c8e8"          # noble gases
for _s in "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split():
    _ELEM_COLORS[_s] = "#a8d4c8"          # lanthanides
for _s in "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No".split():
    _ELEM_COLORS[_s] = "#9dc8b8"          # actinides
_SPECIAL_COLOR = "#b8b8c8"                # TFY, IC, unknown channels

# ── Box geometry shared between selected / deselected states ─────────────────
_BOX_COMMON = (
    "width:52px;height:64px;border-radius:4px;cursor:pointer;user-select:none;"
    "display:flex;flex-direction:column;align-items:center;justify-content:center;"
    "padding:2px 1px;box-sizing:border-box;"
)


def _box_style(bg: str, selected: bool) -> str:
    if selected:
        return f"{_BOX_COMMON}background:{bg};border:2px solid rgba(0,0,0,0.22);"
    return f"{_BOX_COMMON}background:#cccccc;border:2px solid #aaaaaa;opacity:0.35;"


def _parse_channel(name: str) -> tuple:
    """Return (atomic_num | None, display_symbol, subscript, bg_color).

    Examples::
        "Ti"                     → (22, "Ti",  "",  "#aacce8")
        "Ba_L"                   → (56, "Ba",  "L", "#f4a8a8")
        "Total_Fluorescence_Yield" → (None, "TFY", "", "#b8b8c8")
        "us_ic"                  → (None, "IC",  "", "#b8b8c8")
    """
    low = name.lower()
    if "fluorescence_yield" in low:
        return None, "TFY", "", _SPECIAL_COLOR
    if low.endswith("_ic") or low in ("ic", "us_ic"):
        return None, "IC", "", _SPECIAL_COLOR

    parts = name.split("_")
    base = parts[0]
    sub  = parts[1] if len(parts) >= 2 and len(parts[1]) <= 2 else ""

    try:
        import xraylib
        z = xraylib.SymbolToAtomicNumber(base)
        z = z if z > 0 else None
    except Exception:
        z = None

    bg = _ELEM_COLORS.get(base, _SPECIAL_COLOR)
    return z, base, sub, bg


def _default_shell(z: int | None) -> str:
    """K shell for Z < 50; L shell for heavier elements."""
    return "L" if (z is not None and z >= 50) else "K"


def _xraylib_emission(z: int, shell: str) -> float:
    """Return Kα or Lα emission line energy (keV) for atomic number z."""
    try:
        import xraylib
        line = xraylib.KA_LINE if shell == "K" else xraylib.LA_LINE
        return xraylib.LineEnergy(z, line)
    except Exception:
        return 0.0


def _xraylib_density(z: int) -> float:
    """Return elemental density (g/cm³) for atomic number z."""
    try:
        import xraylib
        return xraylib.ElementDensity(z)
    except Exception:
        return 0.0


class _ElementSelector:
    """Periodic-table tile selector for element channels.

    Per-element values exposed as properties (only real elements, i.e. z is not None):
      ``.value``               → comma-sep element names (selected)
      ``.indices_value``       → comma-sep absolute HDF5 channel indices
      ``.shell_value``         → comma-sep K/L shell per selected element
      ``.density_value``       → comma-sep densities (g/cm³) per selected element
      ``.emission_energy_value``→ comma-sep emission energies (keV) per selected element
    """

    def __init__(self):
        self._all_names: list = []
        self._states: dict = {}   # name → state dict

    # ── Value accessors ───────────────────────────────────────────────────────
    @property
    def value(self) -> str:
        return ", ".join(
            n for n, st in self._states.items()
            if st["val"] and st["z"] is not None
        )

    @property
    def indices_value(self) -> str:
        return ", ".join(
            str(i) for i, n in enumerate(self._all_names)
            if self._states.get(n, {}).get("val", False)
            and self._states.get(n, {}).get("z") is not None
        )

    @property
    def shell_value(self) -> str:
        return ", ".join(
            st["shell"] for name, st in self._states.items()
            if st["val"] and st["z"] is not None
        )

    @property
    def density_value(self) -> str:
        vals = []
        for name, st in self._states.items():
            if not (st["val"] and st["z"] is not None):
                continue
            inp = st.get("density_inp")
            v = inp.value if inp is not None else 0.0
            vals.append(f"{v:.4g}" if v else "0")
        return ", ".join(vals)

    @property
    def emission_energy_value(self) -> str:
        vals = []
        for name, st in self._states.items():
            if not (st["val"] and st["z"] is not None):
                continue
            inp = st.get("emission_inp")
            v = inp.value if inp is not None else 0.0
            vals.append(f"{v:.4g}" if v else "0")
        return ", ".join(vals)

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _refresh(state: dict) -> None:
        state["box"].style(replace=_box_style(state["bg"], state["val"]))

    def _make_toggle(self, state: dict):
        def toggle():
            state["val"] = not state["val"]
            self._refresh(state)
        return toggle

    @staticmethod
    def _apply_shell_style(state: dict) -> None:
        active = state["shell"]
        for btn, name in [(state.get("k_btn"), "K"), (state.get("l_btn"), "L")]:
            if btn is None:
                continue
            if name == active:
                btn.props(remove="outline color=grey-5", add="color=teal-7 unelevated")
            else:
                btn.props(remove="color=teal-7 unelevated", add="outline color=grey-5")

    def _make_shell_toggle(self, state: dict, shell: str):
        def toggle():
            state["shell"] = shell
            self._apply_shell_style(state)
            # Auto-update emission energy to match new shell
            z = state.get("z")
            e_inp = state.get("emission_inp")
            if z is not None and e_inp is not None:
                em_E = _xraylib_emission(z, shell)
                if em_E:
                    e_inp.set_value(round(em_E, 4))
        return toggle

    # ── Public API ────────────────────────────────────────────────────────────
    def set_value(self, s: str) -> None:
        requested = {x.strip() for x in s.split(",") if x.strip()}
        for name, st in self._states.items():
            new_val = name in requested
            if st["val"] != new_val:
                st["val"] = new_val
                self._refresh(st)

    def select_all(self, selected: bool = True) -> None:
        for st in self._states.values():
            if st["val"] != selected:
                st["val"] = selected
                self._refresh(st)

    def autofill_from_xraylib(self) -> None:
        """Auto-populate density and emission energy from xraylib for all real elements."""
        for name, st in self._states.items():
            z = st.get("z")
            if z is None:
                continue
            shell = st.get("shell", "K")
            em_E = _xraylib_emission(z, shell)
            density = _xraylib_density(z)
            d_inp = st.get("density_inp")
            e_inp = st.get("emission_inp")
            if d_inp is not None and density:
                d_inp.set_value(round(density, 4))
            if e_inp is not None and em_E:
                e_inp.set_value(round(em_E, 4))

    def populate(self, names: list, container) -> None:
        """(Re-)build periodic-table tiles + K/L shell selectors + per-element inputs."""
        self._all_names = list(names)
        self._states.clear()
        container.clear()
        with container:
            for name in names:
                z, sym, sub, bg = _parse_channel(name)
                default_shell = _default_shell(z)
                state: dict = {
                    "val": True, "box": None, "bg": bg, "z": z,
                    "shell": default_shell,
                    "k_btn": None, "l_btn": None,
                    "density_inp": None, "emission_inp": None,
                }
                self._states[name] = state

                with ui.column().classes("items-center").style("gap:4px;"):
                    # ── Periodic-table tile ───────────────────────────────────
                    with ui.element("div").style(
                        _box_style(bg, True)
                    ).on("click", self._make_toggle(state)).classes("select-none") as box:
                        state["box"] = box
                        num_s = (
                            "font-size:9px;color:#333;width:100%;"
                            "text-align:left;padding-left:3px;line-height:1.3;margin:0;"
                        )
                        ui.html(f'<p style="{num_s}">{z if z is not None else ""}</p>')
                        font = "17px" if len(sym) <= 2 else "11px"
                        sym_s = (
                            f"font-size:{font};font-weight:700;color:#111;"
                            "line-height:1.1;text-align:center;margin:0;"
                        )
                        ui.html(f'<p style="{sym_s}">{sym}</p>')
                        if sub:
                            sub_s = (
                                "font-size:10px;color:#444;"
                                "line-height:1.2;text-align:center;margin:0;"
                            )
                            ui.html(f'<p style="{sub_s}">{sub}</p>')

                    # ── K / L shell toggle + per-element inputs (real elements only) ─
                    if z is not None:
                        with ui.row().classes("gap-0"):
                            k_btn = ui.button(
                                "K", on_click=self._make_shell_toggle(state, "K")
                            ).props(
                                "no-caps size=sm "
                                + ("color=teal-7 unelevated" if default_shell == "K"
                                   else "outline color=grey-5")
                            ).style("min-width:26px;")
                            l_btn = ui.button(
                                "L", on_click=self._make_shell_toggle(state, "L")
                            ).props(
                                "no-caps size=sm "
                                + ("color=teal-7 unelevated" if default_shell == "L"
                                   else "outline color=grey-5")
                            ).style("min-width:26px;")
                            state["k_btn"] = k_btn
                            state["l_btn"] = l_btn

                        density_inp = (
                            ui.number(label="ρ (g/cm³)", value=0.0, step=0.001, min=0, format="%.3f")
                            .props("dense outlined")
                            .style("width:96px;")
                            .tooltip(f"{sym} compound density (g/cm³)")
                        )
                        emission_inp = (
                            ui.number(label="E (keV)", value=0.0, step=0.001, min=0, format="%.4f")
                            .props("dense outlined")
                            .style("width:96px;")
                            .tooltip(f"{sym} Kα/Lα emission energy (keV)")
                        )
                        state["density_inp"] = density_inp
                        state["emission_inp"] = emission_inp


class _ICSelector:
    """Dropdown for designating the ion-chamber channel.

    ``.value`` returns the absolute HDF5 channel index as an ``int``
    (compatible with ``_collect_fl_params`` which passes it directly to the backend).
    """

    def __init__(self):
        self._all_names: list = []
        self._select: object = None   # ui.select widget, set by create_widget()

    @property
    def value(self) -> int:
        if self._select is None or not self._all_names:
            return -1
        try:
            return self._all_names.index(self._select.value)
        except (ValueError, TypeError):
            return -1

    def create_widget(self, label: str = "Ion Chamber Channel") -> object:
        self._select = ui.select(label=label, options=[]).props("dense outlined").classes("flex-1")
        return self._select

    def populate(self, names: list) -> None:
        self._all_names = list(names)
        if self._select is not None:
            self._select.options = names
            self._select.value = names[-1] if names else None


class _IndicesProxy:
    """Thin proxy so ``element_channel_indices`` can live in ``input_elements``."""

    def __init__(self, selector: "_ElementSelector"):
        self._sel = selector

    @property
    def value(self) -> str:
        return self._sel.indices_value


class _ShellProxy:
    """Thin proxy so ``xrf_shell`` can live in ``input_elements``."""

    def __init__(self, selector: "_ElementSelector"):
        self._sel = selector

    @property
    def value(self) -> str:
        return self._sel.shell_value


class _DensityProxy:
    """Thin proxy so ``density`` can live in ``input_elements``."""

    def __init__(self, selector: "_ElementSelector"):
        self._sel = selector

    @property
    def value(self) -> str:
        return self._sel.density_value


class _EmissionProxy:
    """Thin proxy so ``emission_energy`` can live in ``input_elements``."""

    def __init__(self, selector: "_ElementSelector"):
        self._sel = selector

    @property
    def value(self) -> str:
        return self._sel.emission_energy_value


class _ValueHolder:
    """Simple value holder for non-widget parameters (e.g. hidden crop coords)."""

    def __init__(self, v):
        self.value = v


def create_fl_parameter_form(state: FLState) -> tuple[dict, list]:
    """Create the FL correction parameter form.

    Returns:
        (input_elements, valid_params) tuple.
    """
    input_elements = {}
    valid_params = []

    with ui.card().classes("w-full"):
        ui.label("FL Correction Parameters").classes("text-lg font-bold mb-2")
        ui.separator()

        # Element + IC selectors — populated when the HDF5 file is loaded.
        elem_selector = _ElementSelector()
        ic_selector   = _ICSelector()
        input_elements["element_type"]            = elem_selector
        input_elements["element_channel_indices"] = _IndicesProxy(elem_selector)
        input_elements["xrf_shell"]               = _ShellProxy(elem_selector)
        input_elements["density"]                 = _DensityProxy(elem_selector)
        input_elements["emission_energy"]         = _EmissionProxy(elem_selector)
        input_elements["ic_channel_idx"]          = ic_selector
        valid_params.extend([
            "element_type", "element_channel_indices", "xrf_shell",
            "density", "emission_energy", "ic_channel_idx",
        ])

        # Mutable refs so on_elements_loaded can reach widgets built after this point.
        _elem_ui = {"container": None, "status": None}

        def on_elements_loaded(names: list) -> None:
            ctr = _elem_ui["container"]
            st  = _elem_ui["status"]
            if ctr is None:
                return
            elem_selector.populate(names, ctr)
            ic_selector.populate(names)
            # Auto-fill densities and emission energies from xraylib immediately.
            elem_selector.autofill_from_xraylib()
            if st is not None:
                st.set_text(f"{len(names)} channel(s) detected — toggle to select elements:")
                st.classes(remove="text-gray-400 italic", add="text-gray-600")

        # --- Data (paths + inspector + physics) ---
        with ui.expansion("Data", icon="folder_open", value=True).classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):

                # ── File paths ────────────────────────────────────────────────
                el = ui.input(
                    "Working Directory (fn_root)",
                    value="/mnt/micdata3/XRF_tomography/FL_correction/FL_correction_new",
                    placeholder="/path/to/working/directory",
                ).classes("w-full font-mono")
                el.tooltip(
                    "Root working directory — contains input files and receives "
                    "all output folders (Angle_prj_*, recon/, mask3D_*.h5)"
                )
                input_elements["fn_root"] = el
                valid_params.append("fn_root")

                el = ui.input(
                    "HDF5 Data File",
                    value="everything.h5",
                    placeholder="everything.h5",
                ).classes("w-full font-mono")
                el.tooltip(
                    "HDF5 file containing XRF data ('data' dataset: "
                    "[n_channels, n_angles, height, width]), 'thetas' dataset"
                )
                input_elements["fn_data"] = el
                valid_params.append("fn_data")

                # ── HDF5 inspector ────────────────────────────────────────────
                # _crop_cb_ref lets the inspector fire on_crop_changed even though
                # the crop inputs are built later in the same expansion section.
                # _pixel_size_ref lets the inspector read the pixel_size_nm widget
                # built further below (ref is filled in after widget creation).
                _crop_cb_ref: list = [None]
                _pixel_size_ref: list = [None]
                create_h5_inspector(
                    input_elements["fn_root"],
                    input_elements["fn_data"],
                    on_elements_loaded=on_elements_loaded,
                    on_crop_changed=lambda x1, y1, x2, y2: (
                        _crop_cb_ref[0](x1, y1, x2, y2) if _crop_cb_ref[0] else None
                    ),
                    pixel_size_nm_ref=_pixel_size_ref,
                )

                ui.separator().classes("my-1")

                # ── Beam / sample parameters ──────────────────────────────────
                ui.label("Sample Physics").classes("text-sm font-semibold text-gray-700 mt-1")
                ui.label(
                    "Select elements and shells from the tiles below. "
                    "Density (ρ) and emission energy (E) are auto-filled from xraylib "
                    "and can be overridden manually."
                ).classes("text-xs text-gray-500 italic")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "X-ray Energy (keV)", value=13.577, step=0.01, min=0.1,
                        format="%.3f",
                    ).classes("flex-1")
                    el.tooltip("Incident beam energy in keV — used to compute fluorescence cross-sections")
                    input_elements["x_ray_energy"] = el
                    valid_params.append("x_ray_energy")

                    el = ui.number(
                        "Pixel Size (nm)", value=500, step=10, min=1,
                    ).classes("flex-1")
                    el.tooltip("Voxel size in nm — converted to cm for attenuation path integrals")
                    input_elements["pixel_size_nm"] = el
                    valid_params.append("pixel_size_nm")
                    _pixel_size_ref[0] = el  # expose to h5_inspector ruler

                # ── Element selector (auto-populated from HDF5) ──────────────
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.label("Elements").classes("text-sm font-medium text-gray-700")
                    ui.space()
                    ui.button(
                        "All",
                        on_click=lambda: elem_selector.select_all(True),
                    ).props("dense flat no-caps size=xs color=primary")
                    ui.button(
                        "None",
                        on_click=lambda: elem_selector.select_all(False),
                    ).props("dense flat no-caps size=xs color=grey")

                status_lbl = ui.label(
                    "Load HDF5 file above to auto-detect element names."
                ).classes("text-xs text-gray-400 italic w-full")
                _elem_ui["status"] = status_lbl

                # ── Element category colour legend ────────────────────────────
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
                # (input_elements registered above)

                # ── Ion chamber channel ───────────────────────────────────────
                with ui.row().classes("w-full gap-4 items-end mt-2"):
                    ic_selector.create_widget("Ion Chamber Channel").tooltip(
                        "Channel used for incident flux normalization. "
                        "Populated automatically from the HDF5 file."
                    )

                # ── Crop region (set via Crop tool in image viewer above) ─────
                # Values are stored in hidden holders updated by the interactive
                # crop tool; no number inputs are shown to the user.
                _crop_x1 = _ValueHolder(0)
                _crop_y1 = _ValueHolder(0)
                _crop_x2 = _ValueHolder(-1)
                _crop_y2 = _ValueHolder(-1)
                input_elements["crop_x_start"] = _crop_x1
                input_elements["crop_x_end"]   = _crop_x2
                input_elements["crop_y_start"] = _crop_y1
                input_elements["crop_y_end"]   = _crop_y2
                valid_params.extend(["crop_x_start", "crop_x_end", "crop_y_start", "crop_y_end"])

                def on_crop_changed(x1: int, y1: int, x2: int, y2: int) -> None:
                    _crop_x1.value = x1
                    _crop_y1.value = y1
                    _crop_x2.value = x2
                    _crop_y2.value = y2

                _crop_cb_ref[0] = on_crop_changed

        # --- Reconstruction ---
        with ui.expansion("Reconstruction", icon="settings", value=True).classes("w-full"):
            with ui.column().classes("w-full gap-4 p-2"):

                # ── Processing ────────────────────────────────────────────────
                ui.label("Processing").classes("text-sm font-semibold text-gray-700")
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Binning Factor", value=4, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip(
                        "Spatial binning factor applied to both projections and "
                        "reconstructions before attenuation calculation"
                    )
                    input_elements["binning_factor"] = el
                    valid_params.append("binning_factor")

                    el = ui.number(
                        "Unit Scale", value=1e15, step=1e14, min=1,
                        format="%.2e"
                    ).classes("flex-1")
                    el.tooltip("Scaling factor applied to projections (molar → femto-molar = 1e15)")
                    input_elements["scale"] = el
                    valid_params.append("scale")

                ui.separator().classes("my-1")

                # ── Detector Mask Geometry ────────────────────────────────────
                ui.label("Detector Mask Geometry").classes("text-sm font-semibold text-gray-700")
                ui.label(
                    "If mask3D_N.h5 already exists in the working directory, it will be "
                    "loaded directly (skipping generation)."
                ).classes("text-xs text-gray-500 italic")
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Horizontal Angle (°)", value=20.6, step=0.1, min=0.1,
                        format="%.1f"
                    ).classes("flex-1")
                    el.tooltip("Horizontal half-angle of the detector solid angle (degrees)")
                    input_elements["det_alfa"] = el
                    valid_params.append("det_alfa")

                    el = ui.number(
                        "Vertical Angle (°)", value=20.6, step=0.1, min=0.1,
                        format="%.1f"
                    ).classes("flex-1")
                    el.tooltip("Vertical half-angle of the detector solid angle (degrees)")
                    input_elements["det_theta"] = el
                    valid_params.append("det_theta")

                    el = ui.number(
                        "Mask Length (px)", value=200, step=10, min=10
                    ).classes("flex-1")
                    el.tooltip(
                        "Maximum radial length for the 3D detector mask. "
                        "File will be saved as mask3D_N.h5."
                    )
                    input_elements["mask_length_maximum"] = el
                    valid_params.append("mask_length_maximum")

                ui.separator().classes("my-1")

                # ── Initial Reconstruction ────────────────────────────────────
                ui.label("Initial Reconstruction (ASTRA)").classes("text-sm font-semibold text-gray-700")
                with ui.row().classes("w-full gap-4"):
                    el = ui.select(
                        label="Method",
                        options=["EM_CUDA", "FBP_CUDA", "SIRT_CUDA", "FBP"],
                        value="EM_CUDA",
                    ).classes("flex-1")
                    el.tooltip("ASTRA tomography reconstruction algorithm")
                    input_elements["recon_method"] = el
                    valid_params.append("recon_method")

                    el = ui.number(
                        "Iterations", value=16, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("Number of iterations for the initial MLEM/SIRT reconstruction")
                    input_elements["recon_n_iter"] = el
                    valid_params.append("recon_n_iter")

                ui.separator().classes("my-1")

                # ── Iterative Correction ──────────────────────────────────────
                ui.label("Iterative Correction").classes("text-sm font-semibold text-gray-700")
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Correction Iters", value=4, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("Number of FL self-absorption correction iterations")
                    input_elements["n_correction_iters"] = el
                    valid_params.append("n_correction_iters")

                    el = ui.number(
                        "MLEM Iters / Step", value=16, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("MLEM iterations per element per correction step")
                    input_elements["correction_n_iter"] = el
                    valid_params.append("correction_n_iter")

                    el = ui.number(
                        "Border Pixels", value=5, step=1, min=0
                    ).classes("flex-1")
                    el.tooltip("Zero out this many pixels at each edge of the reconstructed volume")
                    input_elements["border_pixels"] = el
                    valid_params.append("border_pixels")

                    el = ui.number(
                        "Smooth Filter", value=3, step=2, min=1
                    ).classes("flex-1")
                    el.tooltip("Median filter kernel size applied between iterations")
                    input_elements["smooth_filter_size"] = el
                    valid_params.append("smooth_filter_size")

                ui.separator().classes("my-1")

                # ── Compute Settings ──────────────────────────────────────────
                ui.label("Compute Settings").classes("text-sm font-semibold text-gray-700")
                with ui.row().classes("w-full gap-8 items-center"):
                    el = ui.switch("Use GPU (CUDA)", value=True)
                    el.tooltip(
                        "Use CUDA-accelerated absorption correction. "
                        "Requires numba and a compatible GPU."
                    )
                    input_elements["use_gpu"] = el
                    valid_params.append("use_gpu")

                    el = ui.number("CPU Cores", value=8, step=1, min=1).classes("flex-1")
                    el.tooltip("Number of CPU cores for parallel attenuation computation")
                    input_elements["num_cpu"] = el
                    valid_params.append("num_cpu")

    return input_elements, valid_params
