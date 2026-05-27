"""Parse structured JSON from LLM output."""

import json
import re
import logging

logger = logging.getLogger(__name__)


class AgentOutput:
    """Parsed agent decision."""

    def __init__(self, thought: str = "", speech: str = "",
                 action: dict | None = None):
        self.thought = thought
        self.speech = speech
        self.action = action or {}


def parse_llm_response(response: dict) -> AgentOutput:
    """Extract AgentOutput from an OpenAI-compatible chat completion response.

    Expected LLM output format:
    {
        "thought": "推理过程...",
        "speech": "发言内容...",
        "action": {"type": "vote", "target": 3}
    }
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.warning("Unexpected API response structure: %s",
                       json.dumps(response, ensure_ascii=False)[:500])
        return AgentOutput()

    if not content:
        return AgentOutput()

    # Try to extract JSON from the content
    json_str = _extract_json(content)
    if json_str is None:
        # Treat the whole content as speech (no structured action)
        return AgentOutput(speech=content.strip())

    try:
        data = json.loads(json_str)
        return AgentOutput(
            thought=data.get("thought", ""),
            speech=data.get("speech", ""),
            action=data.get("action", {}),
        )
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from: %s", content[:300])
        return AgentOutput(speech=content.strip())


def _extract_json(text: str) -> str | None:
    """Extract JSON string from text that may contain markdown fences."""
    # Try markdown code block first
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try bare JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return m.group(0).strip()

    return None
