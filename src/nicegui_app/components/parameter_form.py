"""XRF simulation parameter form component."""

from nicegui import ui
from ..state import AppState


def create_parameter_form(
    state: AppState, pre_sections_callback=None
) -> tuple[dict, list]:
    """Create the XRF simulation parameter form.

    Args:
        state: Shared application state.
        pre_sections_callback: Optional callable invoked before form sections
            (used to inject the chat assistant).

    Returns:
        (input_elements, valid_params) tuple where input_elements maps
        parameter names to NiceGUI elements, and valid_params is a list
        of valid parameter names.
    """
    input_elements = {}
    valid_params = []

    with ui.card().classes("w-full"):
        ui.label("Configuration Parameters").classes("text-lg font-bold mb-2")

        # Inject chat assistant before form sections
        if pre_sections_callback:
            pre_sections_callback()

        ui.separator()

        # --- Sample & Experiment ---
        with ui.expansion("Sample & Experiment", icon="science").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                el = ui.input(
                    "Ground Truth File (.npy)",
                    value="/mnt/micdata3/XRF_tomography/simulation/copper_structure/copper_structure_0227.npy",
                    placeholder="/path/to/ground_truth.npy",
                ).classes("w-full font-mono")
                el.tooltip("Full path to the 3D concentration grid in .npy format")
                input_elements["ground_truth_file"] = el
                valid_params.append("ground_truth_file")

                el = ui.input(
                    "Elements (comma-separated)",
                    value="Cu, Ca",
                    placeholder="Cu, Ca, Fe",
                ).classes("w-full")
                el.tooltip("Chemical symbols to simulate, separated by commas")
                input_elements["elements"] = el
                valid_params.append("elements")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Beam Energy (keV)", value=10.0, step=0.1, min=0.1
                    ).classes("flex-1")
                    el.tooltip("Probe beam energy in keV")
                    input_elements["probe_energy"] = el
                    valid_params.append("probe_energy")

                    el = ui.number(
                        "Incident Intensity", value=1.0e7, step=1e6, min=1
                    ).classes("flex-1")
                    el.tooltip("Incident probe intensity (photons/s)")
                    input_elements["incident_probe_intensity"] = el
                    valid_params.append("incident_probe_intensity")

        # --- Physics Model ---
        with ui.expansion("Physics Model", icon="functions").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-8"):
                    el = ui.switch("Model Self-Absorption", value=False)
                    el.tooltip("Include self-absorption in the forward model")
                    input_elements["model_self_absorption"] = el
                    valid_params.append("model_self_absorption")

                    el = ui.switch("Model Probe Attenuation", value=True)
                    el.tooltip("Include probe attenuation along the beam path")
                    input_elements["model_probe_attenuation"] = el
                    valid_params.append("model_probe_attenuation")

        # --- Geometry ---
        with ui.expansion("Geometry", icon="straighten").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Sample Size (cm)", value=0.01, step=0.001, min=0.0001,
                        format="%.4f"
                    ).classes("flex-1")
                    el.tooltip("Physical size of the sample volume in cm")
                    input_elements["sample_size_cm"] = el
                    valid_params.append("sample_size_cm")

                    el = ui.number(
                        "Detector Diameter (cm)", value=0.9, step=0.1, min=0.01
                    ).classes("flex-1")
                    el.tooltip("Diameter of the detector in cm")
                    input_elements["det_size_cm"] = el
                    valid_params.append("det_size_cm")

                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "Sample-Detector Distance (cm)", value=1.5, step=0.1, min=0.01
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

                el = ui.input(
                    "Rotation Angles [x, y, z] (degrees)",
                    value="0.0, 60.0, 10.0",
                    placeholder="0.0, 0.0, 0.0",
                ).classes("w-full")
                el.tooltip("3D rotation angles as comma-separated values: x, y, z (degrees)")
                input_elements["rotation_angles"] = el
                valid_params.append("rotation_angles")

        # --- Compute Settings ---
        with ui.expansion("Compute Settings", icon="memory").classes("w-full"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("w-full gap-4"):
                    el = ui.number(
                        "GPU ID", value=3, step=1, min=-1
                    ).classes("flex-1")
                    el.tooltip("GPU device ID (-1 for CPU)")
                    input_elements["gpu_id"] = el
                    valid_params.append("gpu_id")

                    el = ui.switch("Debug Mode", value=False)
                    el.tooltip("Enable verbose debug output")
                    input_elements["debug"] = el
                    valid_params.append("debug")

                el = ui.input(
                    "Output Suffix", value="", placeholder="optional suffix"
                ).classes("w-full")
                el.tooltip("Optional suffix appended to output filenames")
                input_elements["suffix"] = el
                valid_params.append("suffix")

    return input_elements, valid_params
