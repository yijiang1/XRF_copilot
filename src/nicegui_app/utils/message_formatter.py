"""Message formatting utilities for the status log display."""

import datetime
import re
from html import escape as html_escape

# Matches timestamps like "2026-02-12 11:32:48,852 - worker - INFO - "
_WORKER_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d+\s*-\s*\w+\s*-\s*\w+\s*-\s*"
)


def format_markdown_message(message, level="INFO"):
    """Format a message with timestamp and styled level for display."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Strip embedded timestamps from worker log lines
    message = _WORKER_TS_RE.sub("", message)

    level_colors = {
        "INFO": "#2563eb",
        "WARNING": "#d97706",
        "ERROR": "#dc2626",
        "SUCCESS": "#16a34a",
        "DEBUG": "#6b7280",
        "WORKER": "#7c3aed",
    }
    color = level_colors.get(level, level_colors["INFO"])

    return (
        f'[{html_escape(timestamp)}] '
        f'<strong style="color:{color}">[{html_escape(level)}]</strong>: '
        f'{html_escape(message)}'
    )


def append_to_message_list(message_list, message, level="INFO"):
    """Append a new formatted message to the message list."""
    formatted_message = format_markdown_message(message, level)
    message_list.append(formatted_message)
    return message_list
