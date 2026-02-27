"""Unified file browser dialog for NiceGUI.

Provides a single reusable component for browsing server-side directories
and files, with modes for directory selection, file opening, and file saving.
All filesystem operations run in a background thread with a timeout so that
stale NFS mounts cannot block the NiceGUI event loop.
"""

from __future__ import annotations

import asyncio as _aio
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from nicegui import ui


class BrowseMode(Enum):
    """The three operating modes for the file browser."""

    DIRECTORY = "directory"
    OPEN_FILE = "open_file"
    SAVE_FILE = "save_file"


@dataclass
class BrowseConfig:
    """Configuration for a file browser dialog invocation."""

    mode: BrowseMode = BrowseMode.DIRECTORY
    title: str = ""
    icon: str = ""
    icon_color: str = ""
    browse_roots: list[str] | None = None
    home_path: str = ""
    file_extensions: set[str] | None = None
    allow_dotfiles: set[str] = field(default_factory=set)
    allow_folder_creation: bool = False
    default_filename: str = ""
    start_path: str = ""


_FS_TIMEOUT = 5.0

# Remember the last-visited directory per dialog title across invocations
# within the same server session.
_last_paths: dict[str, str] = {}

_FILE_BROWSER_CSS = """
.fb-item:hover { background: rgba(99, 102, 241, 0.08) !important; }
"""
_css_injected = False


def _inject_css():
    global _css_injected
    if not _css_injected:
        ui.add_css(_FILE_BROWSER_CSS)
        _css_injected = True


def _defaults(cfg: BrowseConfig) -> None:
    """Fill in empty title / icon / icon_color / home_path from mode."""
    if not cfg.title:
        cfg.title = {
            BrowseMode.DIRECTORY: "Browse Directory",
            BrowseMode.OPEN_FILE: "Open File",
            BrowseMode.SAVE_FILE: "Save File",
        }[cfg.mode]
    if not cfg.icon:
        cfg.icon = {
            BrowseMode.DIRECTORY: "folder_open",
            BrowseMode.OPEN_FILE: "file_open",
            BrowseMode.SAVE_FILE: "save",
        }[cfg.mode]
    if not cfg.icon_color:
        cfg.icon_color = "text-indigo-500"
    if not cfg.home_path:
        cfg.home_path = str(Path.home())


async def open_file_browser(
    config: BrowseConfig,
    *,
    target_input: "ui.input | None" = None,
    on_select: Callable[[str], None] | None = None,
) -> str | None:
    """Open a file browser dialog.

    Args:
        config: Controls mode, filters, appearance, and allowed paths.
        target_input: If provided, ``target_input.value`` is set on selection.
        on_select: If provided, called with the selected path string.

    Returns:
        The selected path, or ``None`` if the dialog was cancelled.
    """
    _inject_css()
    _defaults(config)

    # -- filesystem helpers (run in thread) ----------------------------------

    def _list_dir_sync(directory: str) -> tuple[bool, list[str], list[str]]:
        try:
            real = os.path.realpath(directory)
            if not os.path.isdir(real):
                return False, [], []
            if config.browse_roots is not None:
                if not any(real == r or real.startswith(r.rstrip("/") + "/") for r in config.browse_roots):
                    return False, [], []
            entries = sorted(os.listdir(directory))
            subdirs = [
                e
                for e in entries
                if os.path.isdir(os.path.join(directory, e))
                and (not e.startswith(".") or e in config.allow_dotfiles)
            ]
            files: list[str] = []
            if config.mode != BrowseMode.DIRECTORY:
                for e in entries:
                    if e.startswith(".") and e not in config.allow_dotfiles:
                        continue
                    full = os.path.join(directory, e)
                    if not os.path.isfile(full):
                        continue
                    if config.file_extensions is not None:
                        ext = os.path.splitext(e)[1].lower()
                        if ext not in config.file_extensions:
                            continue
                    files.append(e)
            return True, subdirs, files
        except (PermissionError, OSError):
            return False, [], []

    async def _safe_list(directory: str) -> tuple[bool, list[str], list[str]]:
        try:
            return await _aio.wait_for(
                _aio.to_thread(_list_dir_sync, directory),
                timeout=_FS_TIMEOUT,
            )
        except Exception:
            return False, [], []

    # -- determine starting path ---------------------------------------------

    _mem_key = config.title  # unique per browser type
    # Remembered path takes priority; fall back to caller hint then input value.
    raw_start = (
        _last_paths.get(_mem_key, "")
        or config.start_path
        or (target_input.value.strip() if target_input is not None else "")
    )
    start = config.home_path
    if raw_start:
        check = raw_start
        if os.path.isfile(check):
            check = os.path.dirname(check)
        ok, _, _ = await _safe_list(check)
        if ok:
            start = check

    current = [start]

    # -- build dialog --------------------------------------------------------

    with ui.dialog() as dlg, ui.card().style(
        "border-radius: 14px; padding: 20px; min-width: 540px;"
    ):
        # header
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon(config.icon, size="md").classes(config.icon_color)
            ui.label(config.title).classes("text-lg font-semibold")

        # path display (editable with Go button in all modes)
        with ui.row().classes("w-full items-end gap-1"):
            path_display = (
                ui.input("Directory", value=current[0])
                .classes("flex-grow")
                .props("outlined dense")
                .on("keydown.enter", lambda: _nav(path_display.value.strip()))
            )
            ui.button(
                icon="arrow_forward",
                on_click=lambda: _nav(path_display.value.strip()),
            ).props("flat dense").tooltip("Go to path")

        # navigation buttons
        with ui.row().classes("gap-1 my-2"):

            async def _go_up():
                parent = str(Path(current[0]).parent)
                ok, _, _ = await _safe_list(parent)
                if ok:
                    await _nav(parent)
                else:
                    ui.notify(
                        "Cannot navigate above allowed roots", type="warning"
                    )

            ui.button(icon="arrow_upward", on_click=_go_up).props(
                "flat dense size=sm"
            ).tooltip("Parent directory")
            ui.button(
                icon="home",
                on_click=lambda: _nav(config.home_path),
            ).props("flat dense size=sm").tooltip(config.home_path)

            if config.allow_folder_creation:

                def _new_folder():
                    with ui.dialog() as nd, ui.card().style(
                        "border-radius: 12px; padding: 16px; min-width: 320px;"
                    ):
                        ui.label("New Folder").classes("text-base font-semibold")
                        name_input = (
                            ui.input("Folder name")
                            .classes("w-full")
                            .props("outlined dense")
                        )
                        with ui.row().classes("w-full justify-end gap-2 mt-3"):
                            ui.button("Cancel", on_click=nd.close).props(
                                "flat no-caps"
                            )

                            async def _create():
                                n = name_input.value.strip()
                                if not n:
                                    return
                                p = os.path.join(current[0], n)
                                try:
                                    os.makedirs(p, exist_ok=True)
                                    nd.close()
                                    await _nav(p)
                                except OSError as e:
                                    ui.notify(
                                        f"Failed: {e}", type="negative"
                                    )

                            ui.button(
                                "Create", icon="create_new_folder", on_click=_create
                            ).props("unelevated no-caps color=primary")
                    nd.open()

                ui.button(
                    icon="create_new_folder", on_click=_new_folder
                ).props("flat dense size=sm").tooltip("New folder")

        # listing area
        dir_list = ui.column().classes("w-full").style(
            "max-height: 360px; overflow-y: auto; "
            "border: 1px solid #e2e8f0; border-radius: 8px; padding: 4px;"
        )

        # filename input (SAVE_FILE only)
        filename_input = None
        if config.mode == BrowseMode.SAVE_FILE:
            filename_input = (
                ui.input("Filename", value=config.default_filename)
                .classes("w-full mt-2")
                .props("outlined dense")
            )

        # -- navigation function ---------------------------------------------

        async def _nav(new_path):
            ok, subdirs, files = await _safe_list(new_path)
            if not ok:
                ui.notify(
                    "Invalid, restricted, or unresponsive path",
                    type="warning",
                )
                return
            current[0] = new_path
            _last_paths[_mem_key] = new_path
            path_display.value = new_path
            dir_list.clear()
            with dir_list:
                for d in subdirs:
                    full = os.path.join(new_path, d)
                    with ui.row().classes(
                        "w-full items-center px-2 py-1 fb-item rounded"
                        " cursor-pointer"
                    ).on("click", lambda _, p=full: _nav(p)):
                        ui.icon("folder", size="xs").classes("text-amber-500")
                        ui.label(d).classes("text-sm")
                if files:
                    if subdirs:
                        ui.separator().classes("my-1")
                    for f in files:
                        full = os.path.join(new_path, f)
                        if config.mode == BrowseMode.SAVE_FILE:
                            with ui.row().classes(
                                "w-full items-center px-2 py-1 fb-item rounded"
                                " cursor-pointer"
                            ).on(
                                "click",
                                lambda _, e=f: filename_input.set_value(e),
                            ):
                                ui.icon("description", size="xs").classes(
                                    "text-gray-400"
                                )
                                ui.label(f).classes("text-sm text-gray-600")
                        else:
                            with ui.row().classes(
                                "w-full items-center px-2 py-1 fb-item rounded"
                                " cursor-pointer"
                            ).on(
                                "click",
                                lambda _, p=full: _select(p),
                            ):
                                ui.icon("description", size="xs").classes(
                                    config.icon_color
                                )
                                ui.label(f).classes(
                                    "text-sm"
                                )
                if not subdirs and not files:
                    ui.label("(empty)").classes("text-sm text-gray-400 p-2")

        await _nav(current[0])

        # -- action buttons --------------------------------------------------

        def _select(path: str):
            if target_input is not None:
                target_input.set_value(path)
            if on_select is not None:
                on_select(path)
            dlg.submit(path)

        with ui.row().classes("justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=lambda: dlg.submit(None)).props(
                "flat no-caps"
            )

            if config.mode == BrowseMode.DIRECTORY:

                def _select_dir():
                    _select(current[0])

                ui.button(
                    "Select", icon="check", on_click=_select_dir
                ).props("unelevated no-caps color=primary")

            elif config.mode == BrowseMode.SAVE_FILE:

                def _do_save():
                    d = path_display.value.strip()
                    fn = filename_input.value.strip() if filename_input else ""
                    if not d or not fn:
                        ui.notify(
                            "Directory and filename are required",
                            type="negative",
                        )
                        return
                    full_path = os.path.join(d, fn)
                    if os.path.exists(full_path):
                        _confirm_overwrite(full_path)
                    else:
                        _select(full_path)

                def _confirm_overwrite(path: str):
                    with ui.dialog() as confirm, ui.card().style(
                        "border-radius: 12px; padding: 16px; min-width: 360px;"
                    ):
                        ui.label("File already exists").classes(
                            "text-base font-semibold"
                        )
                        ui.label(os.path.basename(path)).classes(
                            "text-sm text-gray-500"
                        )
                        ui.label("Do you want to overwrite it?").classes(
                            "text-sm mt-1"
                        )
                        with ui.row().classes("w-full justify-end gap-2 mt-3"):
                            ui.button(
                                "Cancel", on_click=confirm.close
                            ).props("flat no-caps")
                            ui.button(
                                "Overwrite",
                                icon="save",
                                on_click=lambda: (
                                    confirm.close(), _select(path)
                                ),
                            ).props("unelevated no-caps color=negative")
                    confirm.open()

                ui.button("Save", icon="save", on_click=_do_save).props(
                    "unelevated no-caps color=primary"
                )

    dlg.open()
    result = await dlg
    return result
