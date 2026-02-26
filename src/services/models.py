from pydantic import BaseModel, validator
from typing import List, Optional


class XRFSimulationParams(BaseModel):
    """Pydantic model for XRF simulation parameters."""

    ground_truth_file: str
    probe_energy: float = 10.0
    incident_probe_intensity: float = 1.0e7
    elements: List[str] = ["Cu", "Ca"]
    model_self_absorption: bool = False
    model_probe_attenuation: bool = True
    sample_size_cm: float = 0.01
    det_size_cm: float = 0.9
    det_from_sample_cm: float = 1.5
    det_ds_spacing_cm: float = 0.4
    rotation_angles: List[float] = [0.0, 0.0, 0.0]
    gpu_id: int = 3
    debug: bool = False
    suffix: str = ""

    @validator("ground_truth_file")
    def validate_ground_truth(cls, v):
        if not v.endswith(".npy"):
            raise ValueError("ground_truth_file must end with .npy")
        return v

    @validator("elements", pre=True)
    def validate_elements(cls, v):
        if isinstance(v, str):
            v = [e.strip() for e in v.split(",") if e.strip()]
        return [e.strip() for e in v]

    @validator("rotation_angles")
    def validate_rotation(cls, v):
        if len(v) != 3:
            raise ValueError("rotation_angles must have exactly 3 values")
        return v


class XRFReconstructionParams(BaseModel):
    """Pydantic model for XRF tomographic reconstruction parameters."""

    # ── Data paths ──
    data_path: str
    f_XRF_data: str
    f_XRT_data: str
    recon_path: str
    P_folder: str
    f_P: str = "Intersecting_Length"
    f_recon_grid: str = "grid_concentration"
    f_initial_guess: str = "initialized_grid_concentration"
    f_recon_parameters: str = "recon_parameters.txt"

    # ── Sample geometry ──
    sample_size_n: int = 64
    sample_height_n: int = 64
    sample_size_cm: float = 0.01

    # ── Elements (parsed from user-friendly strings) ──
    # e.g. "Ca, Sc"  →  {"Ca": 20, "Sc": 21}  (atomic numbers auto-looked up via xraylib)
    element_symbols: str = "Ca, Sc"
    # e.g. "Ca K, Ca L, Sc K, Sc L"  →  [["Ca","K"],["Ca","L"],...]
    element_lines_roi_str: str = "Ca K, Ca L, Sc K, Sc L"
    # e.g. "2, 2"  →  [2, 2]
    n_line_group_each_element_str: str = "2, 2"

    # ── Probe ──
    probe_energy: float = 20.0
    probe_intensity: float = 1.0e7
    probe_att: bool = True

    # ── Reconstruction settings ──
    n_epochs: int = 300
    save_every_n_epochs: int = 10
    minibatch_size: int = 64
    lr: float = 1.0e-3
    b1: float = 0.0
    b2: float = 1.0
    selfAb: bool = True

    # ── Initialization ──
    ini_kind: str = "const"
    init_const: float = 0.0
    cont_from_check_point: bool = False
    use_saved_initial_guess: bool = False

    # ── Detector geometry (for simulated data) ──
    manual_det_coord: bool = False
    det_dia_cm: float = 0.9
    det_from_sample_cm: float = 1.6
    det_ds_spacing_cm: float = 0.4
    det_on_which_side: str = "positive"
    manual_det_area: bool = False

    # ── Data indexing ──
    XRT_ratio_dataset_idx: int = 3
    scaler_counts_us_ic_dataset_idx: int = 1
    scaler_counts_ds_ic_dataset_idx: int = 2
    theta_ls_dataset: str = "exchange/theta"
    channel_names: str = "exchange/elements"

    # ── Compute ──
    gpu_id: int = 3


class DiReconParams(BaseModel):
    """Pydantic model for Di et al. 2017 XRF tomographic reconstruction (Python/PyTorch)."""

    # ── Data paths ──
    data_path: str
    f_XRF_data: str
    f_XRT_data: str
    recon_path: str
    P_folder: str
    f_P: str = "Intersecting_Length"          # shared with Panpan if geometry matches
    f_recon_grid: str = "di_grid_concentration"
    f_initial_guess: str = "di_initialized_grid_concentration"
    f_recon_parameters: str = "di_recon_parameters.txt"

    # ── Sample geometry ──
    sample_size_n: int = 64
    sample_height_n: int = 64
    sample_size_cm: float = 0.01

    # ── Elements (same format as Panpan) ──
    element_symbols: str = "Ca, Sc"
    element_lines_roi_str: str = "Ca K, Ca L, Sc K, Sc L"
    n_line_group_each_element_str: str = "2, 2"

    # ── Probe ──
    probe_energy: float = 20.0
    probe_intensity: float = 1.0e7
    probe_att: bool = True

    # ── Di et al.-specific optimization settings ──
    loss_type: str = "poisson"       # "poisson" (Poisson NLL) or "ls" (MSE)
    beta1_xrt: float = 1.0           # XRT fidelity weight in joint loss
    tikhonov_lambda: float = 0.0     # L2 regularization on W (0 = off)
    n_outer_epochs: int = 5          # outer bi-level iterations
    lbfgs_n_iter: int = 20           # inner L-BFGS max function evaluations per step
    lbfgs_history: int = 10          # L-BFGS memory size
    minibatch_size: int = 64         # Z-rows per batch (same meaning as Panpan)
    save_every_n_epochs: int = 1     # checkpoint every N outer epochs
    selfAb: bool = True

    # ── Initialization ──
    ini_kind: str = "const"
    init_const: float = 0.0
    cont_from_check_point: bool = False
    use_saved_initial_guess: bool = False

    # ── Detector geometry ──
    manual_det_coord: bool = False
    det_dia_cm: float = 0.9
    det_from_sample_cm: float = 1.6
    det_ds_spacing_cm: float = 0.4
    det_on_which_side: str = "positive"
    manual_det_area: bool = False

    # ── Data indexing ──
    XRT_ratio_dataset_idx: int = 3
    scaler_counts_us_ic_dataset_idx: int = 1
    scaler_counts_ds_ic_dataset_idx: int = 2
    theta_ls_dataset: str = "exchange/theta"
    channel_names: str = "exchange/elements"

    # ── Compute ──
    gpu_id: int = 3


class FLCorrectionParams(BaseModel):
    """Pydantic model for BNL FL (fluorescence) self-absorption correction."""

    # ── Data paths ──
    fn_root: str          # Working directory (contains data, param, output folders)
    fn_data: str = "everything.h5"    # HDF5 data file (relative to fn_root or abs)
    fn_param: str = "param.txt"       # Parameter file (relative to fn_root or abs)
    theta_ls_dataset: str = "thetas"  # HDF5 dataset key containing rotation angles (degrees)

    # ── Sample physics (from GUI element tiles) ──
    element_symbols: str = ""         # Comma-sep element symbols, e.g. "Ti, Fe, Cu"
    xrf_shell: str = ""              # Comma-sep K/L shell per element, e.g. "K, K, L"
    density: str = ""                # Comma-sep compound densities (g/cm³), e.g. "4.506, 7.874"
    emission_energy: str = ""        # Comma-sep Kα/Lα energies (keV), e.g. "4.509, 6.399"
    probe_energy: float = 13.577      # Incident beam energy (keV)
    pixel_size_nm: float = 500.0     # Voxel size (nm)

    # ── Data selection ──
    element_channel_indices: str = ""  # Absolute HDF5 channel indices for selected elements (comma-sep)
    ic_channel_idx: int = -1           # Absolute ion chamber index in the data array (-1 = last)
    crop_x_start: int = 0            # Crop projection: start x-pixel (0 = no crop)
    crop_x_end: int = -1            # Crop projection: end x-pixel (-1 = no crop)
    crop_y_start: int = 0            # Crop projection: start y-pixel (0 = no crop)
    crop_y_end: int = -1            # Crop projection: end y-pixel (-1 = no crop)

    # ── Processing ──
    binning_factor: int = 4
    scale: float = 1e15              # Unit scale: molar -> femto-molar

    # ── Detector mask geometry ──
    det_alfa: float = 20.6           # Horizontal dispersion angle (degrees)
    det_theta: float = 20.6          # Vertical dispersion angle (degrees)
    mask_length_maximum: int = 200   # Radial mask length (pixels)

    # ── Initial reconstruction ──
    recon_method: str = "EM_CUDA"    # ASTRA method
    recon_n_iter: int = 16           # Iterations for initial MLEM recon

    # ── Iterative correction ──
    n_correction_iters: int = 4      # Number of FL correction iterations
    correction_n_iter: int = 16      # MLEM iterations per correction step
    border_pixels: int = 5           # Edge pixels to zero out
    smooth_filter_size: int = 3      # Median filter kernel size

    # ── Compute ──
    num_cpu: int = 8
    use_gpu: bool = True             # Use CUDA GPU for absorption correction
