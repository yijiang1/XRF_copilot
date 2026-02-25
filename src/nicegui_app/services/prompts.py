"""System prompt construction for the XRF chat assistant."""

from pathlib import Path

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def _load_knowledge(filename: str) -> str:
    """Load a markdown knowledge file from the knowledge directory."""
    return (_KNOWLEDGE_DIR / filename).read_text()


def build_system_prompt(automation_level: str) -> str:
    """Build the LLM system prompt for the XRF simulation assistant.

    Args:
        automation_level: "free", "inquiry", or "suggest"
    """
    param_descriptions = _load_knowledge("param_descriptions.md")

    if automation_level == "free":
        level_instructions = (
            "You are in FREE-FORM mode.\n\n"
            "Do NOT ask the user questions. Wait for the user to provide "
            "instructions in their own words.\n\n"
            "Each time the user sends a message, extract any parameter values "
            "they mention and output them in a fenced ```params block "
            "containing valid JSON. Use exact parameter names from the "
            "AVAILABLE PARAMETERS section. For example:\n\n"
            "```params\n"
            '{"probe_energy": 12.0, "elements": ["Cu", "Fe"]}\n'
            "```\n\n"
            "Only include parameters the user explicitly mentioned or that "
            "can be directly inferred from their instruction. "
            "Briefly acknowledge what you understood and what parameters "
            "you extracted."
        )
    elif automation_level == "inquiry":
        l0_questions = _load_knowledge("l0_questions.md")
        level_instructions = (
            "You are in INQUIRY mode.\n\n"
            "## Instructions\n"
            "Ask ALL questions from the pre-defined PARAMS QUESTIONS below, "
            "one by one.\n"
            "Ask ONLY ONE question at a time, following the pre-defined order.\n"
            "YOU MUST ASK ALL RELEVANT QUESTIONS. "
            "DO NOT SKIP QUESTIONS. DO NOT CHANGE THE ORDER.\n\n"
            "IMPORTANT: Do NOT output any ```params blocks while asking "
            "questions. Remember the user's answers internally as you go.\n\n"
            "Once ALL questions have been asked, output exactly "
            '"COLLECT_COMPLETE" followed by a single ```params block '
            "containing ALL collected values as valid JSON. Use exact "
            "parameter names from the AVAILABLE PARAMETERS section.\n\n"
            "Only include values the user explicitly stated.\n\n"
            "## PARAMS QUESTIONS\n"
            f"{l0_questions}"
        )
    else:
        l0_questions = _load_knowledge("l0_questions.md")
        level_instructions = (
            "You are in SUGGESTION mode.\n\n"
            "## Instructions\n"
            "Ask ALL questions from the pre-defined PARAMS QUESTIONS below, "
            "one by one.\n"
            "Ask ONLY ONE question at a time, following the pre-defined order.\n\n"
            "Once ALL questions have been asked, use your domain knowledge "
            "of X-ray fluorescence to suggest optimal parameter values. "
            "Output exactly "
            '"COLLECT_COMPLETE" followed by a single ```params block '
            "containing ALL suggested values as valid JSON.\n\n"
            "Include both user-stated values and your recommendations.\n\n"
            "## PARAMS QUESTIONS\n"
            f"{l0_questions}"
        )

    return f"""\
You are Fluoro, an XRF fluorescence simulation assistant for synchrotron X-ray experiments.
You help users configure parameters for XRF simulation using the simulate_XRF_maps software.

## Your Role
- Help users set up simulation parameters by asking about their experiment.
- You are embedded in a GUI that has a parameter form.
- Keep your responses concise and focused.

## Automation Level
{level_instructions}

## Available Parameters
{param_descriptions}
"""
