from dataclasses import dataclass, field


@dataclass
class FLState:
    # FL correction status
    is_running: bool = False
    button_status: str = "idle"  # "idle" or "processing"
    button_timestamp: float = 0.0

    # Progress (step-based)
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0
    step_label: str = ""

    # Results
    recon_file: str = ""
    results_ready: bool = False
    results_displayed: bool = False

    # Log messages
    messages: list = field(default_factory=list)

    # Busy flag
    is_busy: bool = False


@dataclass
class ReconState:
    # Reconstruction status
    is_running: bool = False
    button_status: str = "idle"  # "idle" or "processing"
    button_timestamp: float = 0.0

    # Progress (epoch-based)
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0

    # Results
    recon_file: str = ""
    results_ready: bool = False
    results_displayed: bool = False

    # Log messages
    messages: list = field(default_factory=list)

    # Busy flag
    is_busy: bool = False


@dataclass
class AppState:
    # Simulation status
    is_running: bool = False
    button_status: str = "idle"  # "idle" or "processing"
    button_timestamp: float = 0.0

    # Progress
    current_batch: int = 0
    total_batches: int = 0
    progress_percent: float = 0.0

    # Results
    sim_xrf_file: str = ""
    sim_xrt_file: str = ""
    results_ready: bool = False
    results_displayed: bool = False

    # Log messages
    messages: list = field(default_factory=list)

    # Busy flag for image loading
    is_busy: bool = False
