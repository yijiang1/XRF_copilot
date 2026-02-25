"""Async LLM service for the XRF chat assistant."""

import json
import re
from openai import AsyncOpenAI
from ..config import ANL_USERNAME, ARGO_BASE_URL, ARGO_MODEL
from .prompts import build_system_prompt


class ChatAssistantService:
    """Async Argo Gateway wrapper with conversation history and parameter extraction."""

    def __init__(self, automation_level: str = "free"):
        self._client = AsyncOpenAI(
            api_key=ANL_USERNAME,
            base_url=ARGO_BASE_URL,
        )
        self._model = ARGO_MODEL
        self._automation_level = automation_level
        self._messages: list[dict] = []
        self._rebuild_system_prompt()

    def set_automation_level(self, level: str):
        self._automation_level = level
        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self):
        system_msg = {
            "role": "system",
            "content": build_system_prompt(self._automation_level),
        }
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0] = system_msg
        else:
            self._messages.insert(0, system_msg)

    async def send_message(self, user_message: str) -> tuple[str, dict | None]:
        """Send a user message and get the assistant's response.

        Returns:
            (response_text, suggested_params) where suggested_params is
            a dict parsed from ```params blocks, or None.
        """
        self._messages.append({"role": "user", "content": user_message})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            temperature=0.7,
            max_tokens=1024,
        )

        assistant_content = response.choices[0].message.content
        self._messages.append(
            {"role": "assistant", "content": assistant_content}
        )

        suggested_params = self._extract_params(assistant_content)
        return assistant_content, suggested_params

    def _extract_params(self, text: str) -> dict | None:
        """Extract parameter suggestions from fenced code blocks."""
        for pattern in [
            r"```params\s*\n(.*?)\n\s*```",
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    def get_greeting(self) -> str:
        """Return the initial greeting and add it to conversation history."""
        if self._automation_level == "free":
            greeting = (
                "Hello! I'm Fluoro, an AI assistant for XRF fluorescence simulation. "
                "Tell me about your experiment or specify any parameters "
                "you'd like to set, and I'll update the form for you."
            )
        elif self._automation_level == "inquiry":
            greeting = (
                "Hello! I'm Fluoro, an AI assistant for XRF fluorescence simulation. "
                "I'll ask you a series of questions one by one to gather "
                "information about your simulation setup, "
                "and then fill in the parameters for you.\n\n"
                "Let's begin with the first question:\n\n"
                "What is the full path to your ground truth objects file (.npy)?"
            )
        else:
            greeting = (
                "Hello! I'm Fluoro, an AI assistant for XRF fluorescence simulation. "
                "I'll ask you some questions about your experiment, "
                "and then recommend simulation parameters based on your answers.\n\n"
                "Let's begin with the first question:\n\n"
                "What is the full path to your ground truth objects file (.npy)?"
            )
        self._messages.append({"role": "assistant", "content": greeting})
        return greeting

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        system_msg = self._messages[0] if self._messages else None
        self._messages.clear()
        if system_msg:
            self._messages.append(system_msg)
