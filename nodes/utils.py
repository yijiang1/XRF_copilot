from typing import List
from nodeology.state import State
from nodeology.log import log_print_color

def record_messages(
    state: State, messages: List[tuple[str, str, str]], print_to_console: bool = True
):
    """Record messages to state and log them with color.

    Args:
        state: State object to store messages in
        messages: List of (role, message, color) tuples to record
    """

    for role, message, color in messages:
        # Add check for messages key
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append({"role": role, "content": message})
        log_print_color(message, color, print_to_console)
