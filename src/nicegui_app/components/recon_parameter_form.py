"""XRF reconstruction parameter form component."""

from nicegui import ui
from ..state import ReconState


def create_recon_parameter_form(state: ReconState) -> tuple[dict, list]:
    """Create the XRF reconstruction parameter form.

    Returns:
        (input_elements, valid_params) tuple.
    """
    input_elements = {}
    valid_params = []

    with ui.card().classes("w-full"):
        ui.label("Reconstruction Parameters").classes("text-lg font-bold mb-2")
        ui.separator()

        # --- Data & Paths ---
        with ui.expansion("Data & Paths", icon="folder_open").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                el = ui.input(
                    "Data Directory",
                    value="",
                    placeholder="/path/to/data/",
                ).classes("w-full font-mono")
                el.tooltip("Directory containing the HDF5 data files")
                input_elements["data_path"] = el
                valid_params.append("data_path")

                with ui.row().classes("w-full gap-4"):
                    el = ui.input(
                        "XRF Data File", value="", placeholder="xrf_data.h5"
                    ).classes("flex-1 font-mono")
                    el.tooltip("Filename of the XRF measurement HDF5 file (inside data_path)")
                    input_elements["f_XRF_data"] = el
                    valid_params.append("f_XRF_data")

                    el = ui.input(
                        "XRT Data File", value="", placeholder="xrt_data.h5"
                    ).classes("flex-1 font-mono")
                    el.tooltip("Filename of the XRT transmission HDF5 file (inside data_path)")
                    input_elements["f_XRT_data"] = el
                    valid_params.append("f_XRT_data")

                el = ui.input(
                    "Reconstruction Output Directory",
                    value="",
                    placeholder="/path/to/recon/",
                ).classes("w-full font-mono")
                el.tooltip("Directory where reconstruction results will be saved")
                input_elements["recon_path"] = el
                valid_params.append("recon_path")

                with ui.row().classes("w-full gap-4"):
                    el = ui.input(
                        "Projection Matrix Folder", value="", placeholder="/path/to/P/"
                    ).classes("flex-1 font-mono")
                    el.tooltip("Directory where intersection length matrices are stored/cached")
                    input_elements["P_folder"] = el
                    valid_params.append("P_folder")

                    el = ui.input(
                        "Projection Matrix File", value="Intersecting_Length"
                    ).classes("flex-1")
                    el.tooltip("Base filename for the intersection length matrix")
                    input_elements["f_P"] = el
                    valid_params.append("f_P")

                with ui.row().classes("w-full gap-4"):
                    el = ui.input(
                        "Recon Grid File", value="grid_concentration"
                    ).classes("flex-1")
                    el.tooltip("Base filename for the output reconstruction grid")
                    input_elements["f_recon_grid"] = el
                    valid_params.append("f_recon_grid")

                    el = ui.input(
                        "Recon Parameters File", value="recon_parameters.txt"
                    ).classes("flex-1")
                    el.tooltip("Filename to save reconstruction parameters summary")
                    input_elements["f_recon_parameters"] = el
                    valid_params.append("f_recon_parameters")

                el = ui.input(
                    "Initial Guess File", value="initialized_grid_concentration"
                ).classes("w-full")
                el.tooltip("Base filename for the initial concentration grid")
                input_elements["f_initial_guess"] = el
                valid_params.append("f_initial_guess")

        # --- Sample & Probe ---
        with ui.expansion("Sample & Probe", icon="science").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Sample Width (pixels)", value=64, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("Number of pixels along the sample width/depth axis")
                    input_elements["sample_size_n"] = el
                    valid_params.append("sample_size_n")

                    el = ui.number(
                        "Sample Height (pixels)", value=64, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("Number of pixels along the sample height axis")
                    input_elements["sample_height_n"] = el
                    valid_params.append("sample_height_n")

                    el = ui.number(
                        "Sample Size (cm)", value=0.01, step=0.001, min=0.0001,
                        format="%.4f"
                    ).classes("flex-1")
                    el.tooltip("Physical size of the sample in cm")
                    input_elements["sample_size_cm"] = el
                    valid_params.append("sample_size_cm")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Beam Energy (keV)", value=20.0, step=0.5, min=0.1
                    ).classes("flex-1")
                    el.tooltip("Incident X-ray beam energy in keV")
                    input_elements["probe_energy"] = el
                    valid_params.append("probe_energy")

                    el = ui.number(
                        "Probe Intensity", value=1.0e7, step=1e6, min=1
                    ).classes("flex-1")
                    el.tooltip("Incident probe intensity (photons/s)")
                    input_elements["probe_intensity"] = el
                    valid_params.append("probe_intensity")

                el = ui.switch("Model Probe Attenuation", value=True)
                el.tooltip("Account for probe beam attenuation through the sample")
                input_elements["probe_att"] = el
                valid_params.append("probe_att")

        # --- Elements ---
        with ui.expansion("Elements", icon="biotech").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                el = ui.input(
                    "Elements & Atomic Numbers",
                    value="Ca:20, Sc:21",
                    placeholder="Ca:20, Sc:21, Fe:26",
                ).classes("w-full")
                el.tooltip("Element symbol and atomic number pairs, separated by commas")
                input_elements["elements_atomic_numbers"] = el
                valid_params.append("elements_atomic_numbers")

                el = ui.input(
                    "Element Lines ROI",
                    value="Ca K, Ca L, Sc K, Sc L",
                    placeholder="Ca K, Ca L, Sc K",
                ).classes("w-full")
                el.tooltip(
                    "Fluorescence lines to reconstruct. Format: 'Element Line' pairs "
                    "separated by commas (K, L, or M shell)"
                )
                input_elements["element_lines_roi_str"] = el
                valid_params.append("element_lines_roi_str")

                el = ui.input(
                    "Lines per Element",
                    value="2, 2",
                    placeholder="2, 2",
                ).classes("w-full")
                el.tooltip(
                    "Number of ROI lines per element, matching the element order above"
                )
                input_elements["n_line_group_each_element_str"] = el
                valid_params.append("n_line_group_each_element_str")

        # --- Reconstruction Settings ---
        with ui.expansion("Reconstruction Settings", icon="tune").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Epochs", value=300, step=10, min=1
                    ).classes("flex-1")
                    el.tooltip("Total number of reconstruction epochs (iterations)")
                    input_elements["n_epochs"] = el
                    valid_params.append("n_epochs")

                    el = ui.number(
                        "Save Every N Epochs", value=10, step=1, min=1
                    ).classes("flex-1")
                    el.tooltip("Save a checkpoint every N epochs")
                    input_elements["save_every_n_epochs"] = el
                    valid_params.append("save_every_n_epochs")

                    el = ui.number(
                        "Minibatch Size", value=64, step=8, min=1
                    ).classes("flex-1")
                    el.tooltip("Number of projection angles per minibatch")
                    input_elements["minibatch_size"] = el
                    valid_params.append("minibatch_size")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Learning Rate", value=1.0e-3, step=1e-4, min=1e-8,
                        format="%.6f"
                    ).classes("flex-1")
                    el.tooltip("Adam optimizer learning rate")
                    input_elements["lr"] = el
                    valid_params.append("lr")

                    el = ui.number(
                        "Beta 1 (b1)", value=0.0, step=0.01, min=0.0, max=1.0,
                        format="%.3f"
                    ).classes("flex-1")
                    el.tooltip("Adam optimizer beta1 (first moment decay)")
                    input_elements["b1"] = el
                    valid_params.append("b1")

                    el = ui.number(
                        "Beta 2 (b2)", value=1.0, step=0.01, min=0.0, max=1.0,
                        format="%.3f"
                    ).classes("flex-1")
                    el.tooltip("Adam optimizer beta2 (second moment decay)")
                    input_elements["b2"] = el
                    valid_params.append("b2")

                with ui.row().classes("w-full gap-8"):
                    el = ui.switch("Self-Absorption", value=True)
                    el.tooltip("Account for self-absorption of fluorescent X-rays")
                    input_elements["selfAb"] = el
                    valid_params.append("selfAb")

                    el = ui.switch("Continue from Checkpoint", value=False)
                    el.tooltip("Resume reconstruction from a previously saved checkpoint")
                    input_elements["cont_from_check_point"] = el
                    valid_params.append("cont_from_check_point")

                    el = ui.switch("Use Saved Initial Guess", value=False)
                    el.tooltip("Load a previously saved initial concentration grid")
                    input_elements["use_saved_initial_guess"] = el
                    valid_params.append("use_saved_initial_guess")

                with ui.row().classes("w-full gap-4"):
                    el = ui.select(
                        label="Initialization Kind",
                        options=["const", "rand"],
                        value="const",
                    ).classes("flex-1")
                    el.tooltip("How to initialize the concentration grid")
                    input_elements["ini_kind"] = el
                    valid_params.append("ini_kind")

                    el = ui.number(
                        "Initial Constant Value", value=0.0, step=0.01, format="%.4f"
                    ).classes("flex-1")
                    el.tooltip("Initial concentration value when ini_kind='const'")
                    input_elements["init_const"] = el
                    valid_params.append("init_const")

        # --- Detector Geometry ---
        with ui.expansion("Detector Geometry", icon="straighten").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-8"):
                    el = ui.switch("Manual Detector Coordinates", value=False)
                    el.tooltip("Use manually specified detector coordinates instead of auto-computed")
                    input_elements["manual_det_coord"] = el
                    valid_params.append("manual_det_coord")

                    el = ui.switch("Manual Detector Area", value=False)
                    el.tooltip("Use manually specified detector area instead of auto-computed from diameter")
                    input_elements["manual_det_area"] = el
                    valid_params.append("manual_det_area")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Detector Diameter (cm)", value=0.9, step=0.1, min=0.01
                    ).classes("flex-1")
                    el.tooltip("Diameter of the fluorescence detector in cm")
                    input_elements["det_dia_cm"] = el
                    valid_params.append("det_dia_cm")

                    el = ui.number(
                        "Sample-Detector Distance (cm)", value=1.6, step=0.1, min=0.01
                    ).classes("flex-1")
                    el.tooltip("Distance from sample to detector in cm")
                    input_elements["det_from_sample_cm"] = el
                    valid_params.append("det_from_sample_cm")

                    el = ui.number(
                        "Detector Pixel Spacing (cm)", value=0.4, step=0.05, min=0.01
                    ).classes("flex-1")
                    el.tooltip("Spacing between detector pixels in cm")
                    input_elements["det_ds_spacing_cm"] = el
                    valid_params.append("det_ds_spacing_cm")

                el = ui.select(
                    label="Detector Side",
                    options=["positive", "negative"],
                    value="positive",
                ).classes("w-full")
                el.tooltip("Which side of the sample the detector is on (positive = right/top)")
                input_elements["det_on_which_side"] = el
                valid_params.append("det_on_which_side")

        # --- Data Indexing ---
        with ui.expansion("Data Indexing", icon="table_rows").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "XRT Ratio Dataset Index", value=3, step=1, min=0
                    ).classes("flex-1")
                    el.tooltip("HDF5 dataset index for XRT transmission ratio data")
                    input_elements["XRT_ratio_dataset_idx"] = el
                    valid_params.append("XRT_ratio_dataset_idx")

                    el = ui.number(
                        "Upstream IC Dataset Index", value=1, step=1, min=0
                    ).classes("flex-1")
                    el.tooltip("HDF5 dataset index for upstream ion chamber counts")
                    input_elements["scaler_counts_us_ic_dataset_idx"] = el
                    valid_params.append("scaler_counts_us_ic_dataset_idx")

                    el = ui.number(
                        "Downstream IC Dataset Index", value=2, step=1, min=0
                    ).classes("flex-1")
                    el.tooltip("HDF5 dataset index for downstream ion chamber counts")
                    input_elements["scaler_counts_ds_ic_dataset_idx"] = el
                    valid_params.append("scaler_counts_ds_ic_dataset_idx")

                with ui.row().classes("w-full gap-4"):
                    el = ui.input(
                        "Theta Dataset Path", value="exchange/theta"
                    ).classes("flex-1")
                    el.tooltip("HDF5 dataset path for rotation angle array")
                    input_elements["theta_ls_dataset"] = el
                    valid_params.append("theta_ls_dataset")

                    el = ui.input(
                        "Channel Names Dataset Path", value="exchange/elements"
                    ).classes("flex-1")
                    el.tooltip("HDF5 dataset path for element channel names")
                    input_elements["channel_names"] = el
                    valid_params.append("channel_names")

        # --- Compute Settings ---
        with ui.expansion("Compute Settings", icon="memory").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                el = ui.number("GPU ID", value=3, step=1, min=-1).classes("w-full")
                el.tooltip("GPU device ID to use (-1 for CPU)")
                input_elements["gpu_id"] = el
                valid_params.append("gpu_id")

    return input_elements, valid_params
