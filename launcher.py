#!/usr/bin/env python3
"""
XRF Copilot Launcher
================================
A NiceGUI-based launcher that starts the backend (via SSH to a remote
GPU machine) and the frontend (locally) with a shared API key.

Usage:
    python launcher.py
"""

import json
import os
import pty
import secrets
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen, Request

import paramiko
from nicegui import ui, app

PROJECT_ROOT = Path(__file__).resolve().parent

_FAVICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%23b91c1c'/>"
    "<path d='M14,14 L22,14 L32,26 L42,14 L50,14 L38,32 L50,50 L42,50"
    " L32,38 L22,50 L14,50 L26,32 Z' fill='white'/>"
    "</svg>"
)

SETTINGS_FILE = Path.home() / ".xrfcopilot_launcher.json"
LAUNCHER_PORT_START = 8060  # launcher UI runs here

DEFAULT_SETTINGS = {
    "hostname": "refiner.xray.aps.anl.gov",
    "username": os.environ.get("USER", ""),
    "backend_port": 8000,
    "frontend_port": 8050,
    "remote_python": "",
    "remote_dir": f"/mnt/micdata3/XRF_tomography/XRF_copilot",
}

# Per-user overrides — add entries as needed.
USER_DEFAULTS: dict[str, dict] = {
    "yjiang": {
        "hostname": "refiner.xray.aps.anl.gov",
        "remote_python": "/home/beams/YJIANG/anaconda3/envs/xrf_copilot/bin/python",
        "remote_dir": "/mnt/micdata3/XRF_tomography/XRF_copilot",
    },
}

# ── Shared state ─────────────────────────────────────────────────

ssh_client = None
ssh_channel = None
frontend_proc = None
local_backend_proc = None   # used when backend runs locally (no SSH)
su_ssh_proc = None
su_ssh_master_fd = None
api_key = secrets.token_urlsafe(16)


# ── Localhost detection ───────────────────────────────────────────

def _is_local(hostname: str) -> bool:
    """Return True if *hostname* refers to the current machine."""
    if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        local_names = {socket.gethostname().lower(), socket.getfqdn().lower()}
        return hostname.lower() in local_names
    except Exception:
        return False


# ── Settings ─────────────────────────────────────────────────────


def _load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    user_key = settings.get("username", "").lower()
    if user_key in USER_DEFAULTS:
        settings.update(USER_DEFAULTS[user_key])
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        settings.update(saved)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return settings


def _save_settings(hostname, username, backend_port, frontend_port,
                   remote_python, remote_dir):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(
                {
                    "hostname": hostname,
                    "username": username,
                    "backend_port": backend_port,
                    "frontend_port": frontend_port,
                    "remote_python": remote_python,
                    "remote_dir": remote_dir,
                },
                f, indent=2,
            )
    except OSError:
        pass


# ── SSH helpers ──────────────────────────────────────────────────


def _ssh_connect(hostname, username, password):
    """Connect via paramiko with password auth."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=hostname,
        username=username,
        password=password,
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def _stream_channel(channel, log_widget):
    """Read lines from an SSH channel and forward to the log."""
    buf = ""
    try:
        while not channel.closed:
            if channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    log_widget.push(f"[backend] {line}\n")
            elif channel.exit_status_ready():
                break
            else:
                time.sleep(0.2)
        if buf.strip():
            log_widget.push(f"[backend] {buf}\n")
    except Exception as e:
        log_widget.push(f"[backend] SSH stream error: {e}\n")


def _stream_frontend(proc, log_widget):
    """Read lines from the frontend subprocess stdout."""
    try:
        for line in proc.stdout:
            log_widget.push(f"[frontend] {line.rstrip()}\n")
    except Exception:
        pass


def _stream_pty(master_fd, log_widget):
    """Read lines from a pty fd and forward to the log."""
    buf = ""
    try:
        while True:
            r, _, _ = select.select([master_fd], [], [], 1)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        log_widget.push(f"[backend] {line}\n")
            elif su_ssh_proc is not None and su_ssh_proc.poll() is not None:
                break
        if buf.strip():
            log_widget.push(f"[backend] {buf.strip()}\n")
    except Exception as e:
        log_widget.push(f"[backend] Stream error: {e}\n")


def _wait_for_backend(hostname, port, on_ready, on_fail, log_widget,
                      alive_check=None):
    """Poll the backend health endpoint until it responds."""
    if alive_check is None:
        alive_check = lambda: ssh_channel is not None and not ssh_channel.closed
    url = f"http://{hostname}:{port}/"
    for _ in range(30):
        if not alive_check():
            on_fail()
            return
        try:
            resp = urlopen(url, timeout=2)
            if resp.status == 200:
                on_ready()
                return
        except Exception:
            pass
        time.sleep(2)
    log_widget.push("[launcher] ERROR: Backend did not respond within 60 s.\n")
    on_fail()


def _kill_remote_backend(hostname, username, port, password=None):
    """SSH to remote and kill any lingering backend process."""
    try:
        client = _ssh_connect(hostname, username, password)
        client.exec_command(
            f"pkill -f 'src.services.main.*--port {port}'"
        )
        time.sleep(1)
        client.close()
    except Exception:
        pass


# ── UI ───────────────────────────────────────────────────────────


@ui.page("/")
def launcher_page():
    global ssh_client, ssh_channel, frontend_proc, api_key

    settings = _load_settings()

    ui.add_head_html(
        "<style>"
        ".status-dot { display:inline-block; width:12px; height:12px;"
        "  border-radius:50%; margin-right:6px; vertical-align:middle; }"
        ".dot-red   { background:#ef4444; }"
        ".dot-green { background:#22c55e; }"
        ".dot-amber { background:#f59e0b; }"
        "</style>"
        "<script>"
        "new MutationObserver(() => {"
        "  document.querySelectorAll('.q-log__content').forEach(el => {"
        "    el.parentElement.scrollTop = el.parentElement.scrollHeight;"
        "  });"
        "}).observe(document.body, {childList:true, subtree:true});"
        "</script>"
    )

    with ui.column().classes("w-full mx-auto p-6 gap-4"):

        # ── Header ──
        with ui.row().classes("items-center justify-center gap-3 mb-2"):
            ui.icon("biotech", size="2.5rem").classes("text-red-600")
            ui.label("XRF Copilot").classes(
                "text-2xl font-bold text-slate-800"
            )

        # ── Connection Settings ──
        with ui.card().classes("w-full"):
            ui.label("Remote Backend Settings").classes(
                "text-base font-semibold text-slate-700 mb-3"
            )

            hostname = ui.input(
                "Backend Machine (hostname)", value=settings["hostname"]
            ).classes("w-full").props("outlined dense")

            username = ui.input(
                "ANL Username", value=settings["username"]
            ).classes("w-full").props("outlined dense")

            def _on_username_change(e):
                user_key = e.value.strip().lower()
                overrides = USER_DEFAULTS.get(user_key, {})
                if "hostname" in overrides:
                    hostname.set_value(overrides["hostname"])
                if "remote_python" in overrides:
                    remote_python.set_value(overrides["remote_python"])
                if "remote_dir" in overrides:
                    remote_dir.set_value(overrides["remote_dir"])

            username.on_value_change(_on_username_change)

            password = ui.input("Password (not required for localhost)").classes("w-full").props(
                'outlined dense type="password"'
            )
            password.on("keydown.enter", lambda: on_launch())

            with ui.row().classes("w-full gap-4"):
                backend_port = ui.number(
                    "Backend Port", value=settings["backend_port"],
                    step=1, min=1024, max=65535,
                ).classes("flex-1").props("outlined dense")
                frontend_port = ui.number(
                    "Frontend Port", value=settings["frontend_port"],
                    step=1, min=1024, max=65535,
                ).classes("flex-1").props("outlined dense")

            remote_python = ui.input(
                "Remote Python Path", value=settings["remote_python"],
                placeholder="/path/to/envs/xrf_copilot/bin/python",
            ).classes("w-full").props("outlined dense")

            remote_dir = ui.input(
                "Remote Project Directory", value=settings["remote_dir"],
                placeholder="/path/to/XRF_copilot",
            ).classes("w-full").props("outlined dense")

        # ── API Key ──
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("API Key").classes("text-sm font-semibold text-slate-500")
            api_key_input = ui.input(value=api_key).classes("flex-grow").props(
                "outlined dense"
            )

            def _regenerate():
                global api_key
                api_key = secrets.token_urlsafe(16)
                api_key_input.set_value(api_key)

            ui.button(icon="refresh", on_click=_regenerate).props(
                "flat round dense size=sm"
            ).tooltip("Regenerate API key")

        # ── Buttons ──
        with ui.row().classes("w-full justify-center gap-3"):
            launch_btn = ui.button("Launch", icon="rocket_launch").props(
                "color=positive size=lg no-caps"
            )
            open_btn = ui.button("Open App", icon="open_in_new").props(
                "color=primary size=lg no-caps"
            )
            open_btn.disable()
            stop_btn = ui.button("Stop", icon="stop_circle").props(
                "color=warning size=lg no-caps"
            )
            stop_btn.disable()
            quit_btn = ui.button("Quit", icon="power_settings_new").props(
                "color=dark size=lg no-caps"
            )

        # ── Status row ──
        with ui.row().classes("w-full items-center gap-6"):
            with ui.row().classes("items-center"):
                backend_dot = ui.html('<span class="status-dot dot-red"></span>')
                backend_label = ui.label("Backend: Stopped").classes("text-sm")
            with ui.row().classes("items-center"):
                frontend_dot = ui.html('<span class="status-dot dot-red"></span>')
                frontend_label = ui.label("Frontend: Stopped").classes("text-sm")

        # ── Log ──
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("Output Log").classes("text-base font-semibold text-slate-700")

                async def _save_log():
                    content = await ui.run_javascript(
                        "document.querySelector('.q-log__content')?.innerText || ''"
                    )
                    if not content:
                        ui.notify("Log is empty", type="warning")
                        return
                    await ui.run_javascript(f"""
                        const blob = new Blob([{repr(content)}], {{type:'text/plain'}});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url; a.download = 'xrfcopilot_launcher.log';
                        a.click(); URL.revokeObjectURL(url);
                    """)

                ui.button("Save Log", icon="save", on_click=_save_log).props(
                    "flat dense size=sm no-caps"
                )

            log = ui.log(max_lines=300).classes("w-full h-64 font-mono text-xs")

        # ── Kill orphaned backend ──
        kill_btn = ui.button(
            "Kill Orphaned Backend Processes", icon="dangerous"
        ).props("color=negative outline size=md no-caps")

        # ── Helpers ──

        actual_frontend_port = {"value": int(frontend_port.value)}

        def _set_status(which, state):
            """state: 'running' | 'starting' | 'stopped'"""
            dot = backend_dot if which == "backend" else frontend_dot
            lbl = backend_label if which == "backend" else frontend_label
            dot_cls = {"running": "dot-green", "starting": "dot-amber",
                       "stopped": "dot-red"}.get(state, "dot-red")
            text = state.title()
            dot.set_content(f'<span class="status-dot {dot_cls}"></span>')
            lbl.set_text(f"{which.title()}: {text}")

        def _disable_inputs():
            launch_btn.disable()
            open_btn.enable()
            stop_btn.enable()
            for w in [hostname, username, password, backend_port,
                      frontend_port, remote_python, remote_dir]:
                w.disable()

        def _enable_inputs():
            launch_btn.enable()
            open_btn.disable()
            stop_btn.disable()
            for w in [hostname, username, password, backend_port,
                      frontend_port, remote_python, remote_dir]:
                w.enable()

        # ── Launch ──

        def on_launch():
            global ssh_client, ssh_channel, frontend_proc, local_backend_proc, api_key

            h   = hostname.value.strip()
            u   = username.value.strip()
            pw  = password.value or ""
            rpy = remote_python.value.strip()
            rd  = remote_dir.value.strip()
            local = _is_local(h)

            if not h:
                ui.notify("Backend machine is required.", type="negative"); return
            if not local and not u:
                ui.notify("ANL username is required.", type="negative"); return
            if not local and not pw:
                ui.notify("Password is required.", type="negative"); return
            if not rpy:
                ui.notify("Remote Python path is required.", type="negative"); return
            if not rd:
                ui.notify("Remote project directory is required.", type="negative"); return

            try:
                bp = int(backend_port.value)
                fp = int(frontend_port.value)
                if not (1024 <= bp <= 65535) or not (1024 <= fp <= 65535):
                    raise ValueError
            except (ValueError, TypeError):
                ui.notify("Ports must be integers 1024–65535.", type="negative"); return

            api_key = api_key_input.value.strip() or api_key
            _save_settings(h, u, bp, fp, rpy, rd)
            _disable_inputs()

            # Backend command (same for local and remote)
            backend_cmd = [
                rpy, "-m", "src.services.main",
                "--api-key", api_key,
                "--port", str(bp),
                "--host", "0.0.0.0",
            ]

            def on_backend_ready():
                _set_status("backend", "running")
                log.push("[launcher] Backend is ready.\n")
                _start_frontend(h, bp, fp, u)

            def on_backend_fail():
                log.push("[launcher] Launch failed. Check the log above.\n")
                _set_status("backend", "stopped")
                _enable_inputs()

            def do_local_start():
                global local_backend_proc
                log.push(f"[launcher] Starting backend locally in {rd}...\n")
                _set_status("backend", "starting")
                try:
                    local_backend_proc = subprocess.Popen(
                        backend_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        cwd=rd,
                    )
                except Exception as e:
                    log.push(f"[launcher] ERROR: Failed to start local backend: {e}\n")
                    _set_status("backend", "stopped")
                    _enable_inputs()
                    return

                threading.Thread(
                    target=_stream_frontend, args=(local_backend_proc, log), daemon=True
                ).start()

                alive = lambda: local_backend_proc is not None and local_backend_proc.poll() is None
                threading.Thread(
                    target=_wait_for_backend,
                    args=("localhost", bp, on_backend_ready, on_backend_fail, log, alive),
                    daemon=True,
                ).start()

            # Remote command (SSH path): shell string for exec_command / su -c
            remote_cmd = (
                f"cd {shlex.quote(rd)} && "
                f"{rpy} -m src.services.main "
                f"--api-key {shlex.quote(api_key)} --port {bp} --host 0.0.0.0"
            )

            def do_two_step_connect():
                global su_ssh_proc, su_ssh_master_fd
                log.push(
                    f"[launcher] Trying two-step login: su - {u}, then ssh {h}...\n"
                )
                ssh_part = (
                    f"ssh -tt -o StrictHostKeyChecking=no {h} "
                    f"{shlex.quote(remote_cmd)}"
                )
                master_fd, slave_fd = pty.openpty()
                proc = subprocess.Popen(
                    ["su", "-", u, "-c", ssh_part],
                    stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                    preexec_fn=os.setsid,
                )
                os.close(slave_fd)

                output = b""
                deadline = time.time() + 15
                password_sent = False
                while time.time() < deadline:
                    r, _, _ = select.select([master_fd], [], [], 1)
                    if r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            break
                        output += data
                        if b"assword:" in output:
                            os.write(master_fd, (pw + "\n").encode())
                            password_sent = True
                            break

                if not password_sent:
                    log.push("[launcher] ERROR: Two-step login failed (no password prompt).\n")
                    proc.terminate()
                    os.close(master_fd)
                    _enable_inputs()
                    return

                time.sleep(2)
                if proc.poll() is not None:
                    try:
                        remaining = os.read(master_fd, 4096).decode("utf-8", errors="replace")
                    except OSError:
                        remaining = ""
                    log.push(f"[launcher] ERROR: su authentication failed. {remaining}\n")
                    os.close(master_fd)
                    _enable_inputs()
                    return

                su_ssh_proc = proc
                su_ssh_master_fd = master_fd

                log.push(f"[launcher] Authenticated. Starting backend on {h}...\n")
                _set_status("backend", "starting")

                threading.Thread(
                    target=_stream_pty, args=(master_fd, log), daemon=True
                ).start()

                alive = lambda: su_ssh_proc is not None and su_ssh_proc.poll() is None
                threading.Thread(
                    target=_wait_for_backend,
                    args=(h, bp, on_backend_ready, on_backend_fail, log, alive),
                    daemon=True,
                ).start()

            def do_ssh_connect():
                global ssh_client, ssh_channel
                log.push("[launcher] Connecting to backend via SSH...\n")
                try:
                    ssh_client = _ssh_connect(h, u, pw)
                except Exception as e:
                    log.push(f"[launcher] Direct SSH failed: {e}\n")
                    ssh_client = None
                    do_two_step_connect()
                    return

                log.push(f"[launcher] Connected to {h}. Starting backend...\n")
                _set_status("backend", "starting")

                transport = ssh_client.get_transport()
                ssh_channel = transport.open_session()
                ssh_channel.get_pty()
                ssh_channel.exec_command(remote_cmd)

                threading.Thread(
                    target=_stream_channel, args=(ssh_channel, log), daemon=True
                ).start()
                threading.Thread(
                    target=_wait_for_backend,
                    args=(h, bp, on_backend_ready, on_backend_fail, log),
                    daemon=True,
                ).start()

            if local:
                threading.Thread(target=do_local_start, daemon=True).start()
            else:
                threading.Thread(target=do_ssh_connect, daemon=True).start()

        # ── Start frontend ──

        def _start_frontend(h, bp, fp, u):
            global frontend_proc

            actual_fp = _find_free_port(start=fp, max_tries=10)
            if actual_fp != fp:
                log.push(f"[launcher] Port {fp} in use, using {actual_fp}.\n")
                fp = actual_fp
                frontend_port.set_value(fp)
            actual_frontend_port["value"] = fp

            env = os.environ.copy()
            env["API_ENDPOINT"]    = f"http://{h}:{bp}"
            env["BACKEND_API_KEY"] = api_key
            env["HOST"]            = "0.0.0.0"
            env["PORT"]            = str(fp)
            env["ANL_USERNAME"]    = u

            frontend_proc = subprocess.Popen(
                [sys.executable, "-m", "src.nicegui_app", "--api-key", api_key],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            threading.Thread(
                target=_stream_frontend, args=(frontend_proc, log), daemon=True
            ).start()

            _set_status("frontend", "running")
            fqdn = socket.getfqdn()
            app_url = f"http://{fqdn}:{fp}/{api_key}"
            log.push(f"[launcher] App URL: {app_url}\n")

            def _open_when_ready():
                for _ in range(30):
                    if frontend_proc is None or frontend_proc.poll() is not None:
                        return
                    try:
                        resp = urlopen(f"http://localhost:{fp}/", timeout=2)
                        if resp.status == 200:
                            webbrowser.open(app_url)
                            log.push("[launcher] Opened app in browser.\n")
                            return
                    except Exception:
                        pass
                    time.sleep(2)
                log.push("[launcher] Frontend did not respond; open the URL manually.\n")

            threading.Thread(target=_open_when_ready, daemon=True).start()

        # ── Stop ──

        def _stop_simulation_api():
            h  = hostname.value.strip()
            bp = int(backend_port.value)
            url = f"http://{h}:{bp}/stop_simulation/"
            try:
                headers = {"X-API-Key": api_key} if api_key else {}
                req = Request(url, method="POST", headers=headers)
                urlopen(req, timeout=5)
                log.push("[launcher] Stop signal sent to backend.\n")
                time.sleep(1)
            except Exception:
                pass

        def on_stop():
            global ssh_client, ssh_channel, frontend_proc, local_backend_proc
            global su_ssh_proc, su_ssh_master_fd

            log.push("[launcher] Stopping processes...\n")
            _stop_simulation_api()

            # Stop frontend
            if frontend_proc and frontend_proc.poll() is None:
                frontend_proc.terminate()
                try:
                    frontend_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    frontend_proc.kill()
                    frontend_proc.wait()
                log.push("[launcher] Frontend stopped.\n")

            # Stop backend — local mode
            if local_backend_proc is not None:
                if local_backend_proc.poll() is None:
                    local_backend_proc.terminate()
                    try:
                        local_backend_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        local_backend_proc.kill()
                        local_backend_proc.wait()
                local_backend_proc = None
                log.push("[launcher] Local backend stopped.\n")
                _set_status("backend", "stopped")
                _set_status("frontend", "stopped")
                _enable_inputs()
                return

            # Stop backend — two-step mode
            used_two_step = su_ssh_proc is not None
            if su_ssh_proc is not None:
                if su_ssh_proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(su_ssh_proc.pid), signal.SIGTERM)
                        su_ssh_proc.wait(timeout=5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(os.getpgid(su_ssh_proc.pid), signal.SIGKILL)
                            su_ssh_proc.wait()
                        except Exception:
                            pass
                if su_ssh_master_fd is not None:
                    try:
                        os.close(su_ssh_master_fd)
                    except OSError:
                        pass
                su_ssh_proc = None
                su_ssh_master_fd = None
                log.push("[launcher] Backend (su+SSH) stopped.\n")

            # Stop backend — direct SSH mode
            h  = hostname.value.strip()
            u  = username.value.strip()
            bp = int(backend_port.value)
            pw = password.value or None

            if ssh_channel and not ssh_channel.closed:
                ssh_channel.close()
            if ssh_client:
                ssh_client.close()
                log.push("[launcher] Backend SSH closed.\n")

            if not used_two_step:
                _kill_remote_backend(h, u, bp, pw)

            ssh_client = None
            ssh_channel = None
            frontend_proc = None
            _set_status("backend", "stopped")
            _set_status("frontend", "stopped")
            _enable_inputs()

        # ── Open App ──

        def on_open_app():
            fp  = actual_frontend_port["value"]
            key = api_key_input.value
            fqdn = socket.getfqdn()
            ui.navigate.to(f"http://{fqdn}:{fp}/{key}", new_tab=True)

        # ── Kill orphans ──

        def _kill_remote_processes():
            h  = hostname.value.strip()
            u  = username.value.strip()
            pw = password.value or None
            if not h or not u or not pw:
                ui.notify("Hostname, username, and password are required.", type="negative")
                return

            def _kill_via_two_step():
                kill_cmd = (
                    "pkill -f 'src.services.main'; sleep 1; "
                    "pgrep -af 'src.services.main' || echo 'No remaining processes'"
                )
                ssh_part = (
                    f"ssh -tt -o StrictHostKeyChecking=no {h} "
                    f"{shlex.quote(kill_cmd)}"
                )
                master_fd, slave_fd = pty.openpty()
                proc = subprocess.Popen(
                    ["su", "-", u, "-c", ssh_part],
                    stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                    preexec_fn=os.setsid,
                )
                os.close(slave_fd)
                buf = b""
                deadline = time.time() + 15
                password_sent = False
                while time.time() < deadline:
                    r, _, _ = select.select([master_fd], [], [], 1)
                    if r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            break
                        buf += data
                        if b"assword:" in buf:
                            os.write(master_fd, (pw + "\n").encode())
                            password_sent = True
                            break
                if not password_sent:
                    log.push("[launcher] Two-step kill failed.\n")
                    proc.terminate(); os.close(master_fd); return
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                try:
                    r, _, _ = select.select([master_fd], [], [], 1)
                    if r:
                        out = os.read(master_fd, 4096).decode("utf-8", errors="replace").strip()
                        if out:
                            log.push(f"[launcher] {out}\n")
                except OSError:
                    pass
                os.close(master_fd)
                log.push("[launcher] Kill command completed.\n")

            def do_kill():
                log.push(f"[launcher] Killing xrf_copilot backend processes on {h}...\n")
                try:
                    client = _ssh_connect(h, u, pw)
                    _, stdout, _ = client.exec_command(
                        "pkill -f 'src.services.main'; sleep 1; "
                        "pgrep -af 'src.services.main' || echo 'No remaining processes'"
                    )
                    out = stdout.read().decode("utf-8", errors="replace").strip()
                    client.close()
                    if out:
                        log.push(f"[launcher] {out}\n")
                    log.push("[launcher] Kill command completed.\n")
                except Exception as e:
                    log.push(f"[launcher] Direct SSH failed: {e}. Trying su+ssh...\n")
                    try:
                        _kill_via_two_step()
                    except Exception as e2:
                        log.push(f"[launcher] Failed to kill processes: {e2}\n")

            threading.Thread(target=do_kill, daemon=True).start()

        # ── Dialogs ──

        def confirm_stop():
            with ui.dialog() as dlg, ui.card():
                ui.label("Stop all processes?").classes("text-lg font-bold")
                ui.label("This will stop the frontend and backend.").classes("text-sm text-gray-500")
                with ui.row().classes("w-full justify-end gap-2 mt-3"):
                    ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                    ui.button("Stop", on_click=lambda: (dlg.close(), on_stop())).props(
                        "color=warning no-caps"
                    )
            dlg.open()

        def confirm_quit():
            async def do_quit():
                dlg.close()
                on_stop()
                await ui.run_javascript("window.close()")
                app.shutdown()

            with ui.dialog() as dlg, ui.card():
                ui.label("Quit the launcher?").classes("text-lg font-bold")
                ui.label("This will stop all processes and close the launcher.").classes(
                    "text-sm text-gray-500"
                )
                with ui.row().classes("w-full justify-end gap-2 mt-3"):
                    ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                    ui.button("Quit", on_click=do_quit).props("color=dark no-caps")
            dlg.open()

        def confirm_kill():
            with ui.dialog() as dlg, ui.card():
                ui.label("Kill all backend processes?").classes("text-lg font-bold")
                ui.label(
                    "SSH to the backend machine and kill all xrf_copilot processes. "
                    "Use this to clean up orphaned processes after a crash."
                ).classes("text-sm text-gray-500")
                with ui.row().classes("w-full justify-end gap-2 mt-3"):
                    ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                    ui.button("Kill All", on_click=lambda: (dlg.close(), _kill_remote_processes())).props(
                        "color=negative no-caps"
                    )
            dlg.open()

        # Wire up buttons
        launch_btn.on_click(on_launch)
        open_btn.on_click(on_open_app)
        stop_btn.on_click(confirm_stop)
        quit_btn.on_click(confirm_quit)
        kill_btn.on_click(confirm_kill)


# ── Cleanup on exit ───────────────────────────────────────────────


def _cleanup():
    global ssh_client, ssh_channel, frontend_proc, local_backend_proc, su_ssh_proc, su_ssh_master_fd
    if frontend_proc and frontend_proc.poll() is None:
        frontend_proc.terminate()
        try:
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_proc.kill()
    if local_backend_proc is not None and local_backend_proc.poll() is None:
        local_backend_proc.terminate()
        try:
            local_backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            local_backend_proc.kill()
    if su_ssh_proc is not None and su_ssh_proc.poll() is None:
        try:
            os.killpg(os.getpgid(su_ssh_proc.pid), signal.SIGTERM)
            su_ssh_proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(su_ssh_proc.pid), signal.SIGKILL)
            except Exception:
                pass
    if su_ssh_master_fd is not None:
        try:
            os.close(su_ssh_master_fd)
        except OSError:
            pass
    if ssh_channel and not ssh_channel.closed:
        ssh_channel.close()
    if ssh_client:
        ssh_client.close()


app.on_shutdown(_cleanup)


# ── Port detection & run ─────────────────────────────────────────


def _find_free_port(start=LAUNCHER_PORT_START, max_tries=10):
    """Return the first available TCP port starting from *start*."""
    for offset in range(max_tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


launcher_port = _find_free_port()
print(f"Starting launcher on http://localhost:{launcher_port}/")

ui.run(
    title="XRF Copilot Launcher",
    host="0.0.0.0",
    port=launcher_port,
    reload=False,
    show=True,
    favicon=_FAVICON,
)
