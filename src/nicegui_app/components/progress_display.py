"""Progress bar and status info component."""

from nicegui import ui
from ..state import AppState


def create_progress_display(state: AppState) -> tuple[ui.linear_progress, callable]:
    """Create progress bar and info text.

    Returns:
        (progress_bar, update_progress_fn) tuple.
    """
    ui.label("Progress:").classes("font-bold mt-4 mb-1")

    progress_info = ui.label("No simulation running").classes(
        "text-gray-500 text-xl font-semibold w-full mb-1"
    )
    progress_bar = ui.linear_progress(value=0, show_value=False).classes("mb-2")
    progress_label = ui.label("0%").classes("text-center text-sm")

    def update_progress():
        pct = state.progress_percent
        progress_bar.set_value(pct / 100.0)
        progress_label.set_text(f"{pct:.1f}%")

        if state.is_running or state.current_batch > 0:
            info = (
                f"Batch: {state.current_batch}/{state.total_batches}  |  "
                f"Completion: {pct:.1f}%"
            )
            progress_info.set_text(info)
        else:
            progress_info.set_text("No simulation running")

    return progress_bar, update_progress
