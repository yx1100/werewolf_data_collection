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
        self.is_fallback_speech = False


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

    # Multi-strategy parse
    data = _parse_json_robust(json_str)
    if data is None:
        logger.warning("Failed to parse JSON from: %s", content[:300])
        return AgentOutput(speech=content.strip())

    # Defensive unpack: if speech field contains a nested JSON with
    # thought/speech keys, unpack it (LLM sometimes returns double-encoded JSON)
    speech_val = data.get("speech", "")
    if isinstance(speech_val, str) and speech_val.strip().startswith("{"):
        try:
            nested = json.loads(speech_val)
            if isinstance(nested, dict) and "speech" in nested:
                # Merge: nested thought only if outer thought is empty
                if not data.get("thought"):
                    data["thought"] = nested.get("thought", "")
                data["speech"] = nested.get("speech", "")
                if "action" in nested and not data.get("action"):
                    data["action"] = nested["action"]
        except (json.JSONDecodeError, TypeError):
            pass

    # Defensive sanitise: strip LLM self-annotation from speech
    speech_val = data.get("speech", "")
    if isinstance(speech_val, str):
        speech_val = _sanitize_speech(speech_val)

    return AgentOutput(
        thought=data.get("thought", ""),
        speech=speech_val,
        action=data.get("action", {}),
    )


def _parse_json_robust(json_str: str) -> dict | None:
    """Multi-strategy JSON parser for LLM output.

    Tries:
      1. Raw json.loads()
      2. Repair common LLM errors + json.loads()
      3. Regex field extraction from known JSON structure
    """
    # Strategy 1: raw parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 2: repair + parse
    repaired = _repair_json(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 3: regex field extraction
    return _extract_fields(repaired)


def _repair_json(text: str) -> str:
    """Repair common LLM JSON formatting errors.

    Handles:
    - Premature closing brace: {"thought":"..."}, "speech" → {"thought":"...", "speech"
    - Trailing commas: {"a":1,} → {"a":1}
    - Chinese comma as delimiter: ， → ,
    """
    # Fix 1: premature closing brace before known keys
    # LLM sometimes outputs {"thought":"..."}, "speech":"..." → extra } before ,
    text = re.sub(r'"\s*\}\s*,\s*"speech"', '", "speech"', text)
    text = re.sub(r'"\s*\}\s*,\s*"action"', '", "action"', text)

    # Fix 2: trailing comma before } or ]
    text = re.sub(r',\s*\}', '}', text)
    text = re.sub(r',\s*\]', ']', text)

    # Fix 3: Chinese comma as JSON delimiter
    text = text.replace('，', ',')

    return text


def _extract_fields(text: str) -> dict | None:
    """Extract fields from malformed JSON using structural markers.

    Uses the known JSON schema (thought, speech, action keys) as anchors
    to extract field values even when the JSON has syntax errors like
    unescaped inner quotes or missing delimiters.
    """
    result = {}

    # Extract thought: content between "thought":" and the next known boundary
    for end_key in ['"speech"', '"action"', '}']:
        m = re.search(
            r'"thought"\s*:\s*"(.*?)"\s*[,}]\s*' + end_key,
            text, re.DOTALL
        )
        if m:
            result['thought'] = m.group(1)
            break
    if 'thought' not in result:
        # thought-only: {"thought":"..."}
        m = re.search(r'"thought"\s*:\s*"(.*?)"\s*\}', text, re.DOTALL)
        if m:
            result['thought'] = m.group(1)

    # Extract speech: "speech":"CONTENT"[,}]
    m = re.search(r'"speech"\s*:\s*"(.*?)"\s*[,}]', text, re.DOTALL)
    if m:
        result['speech'] = m.group(1)

    # Extract action: "action":{...} or "action":"..."
    m = re.search(r'"action"\s*:\s*(\{.*?\})\s*\}?\s*$', text, re.DOTALL)
    if m:
        try:
            action_text = _repair_json(m.group(1))
            result['action'] = json.loads(action_text)
        except (json.JSONDecodeError, TypeError):
            result['action'] = {}
    else:
        # action as string: "action":"vote"
        m2 = re.search(r'"action"\s*:\s*"([^"]*)"', text)
        if m2:
            result['action'] = {"type": m2.group(1)}

    return result if result else None


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


# ── Speech sanitisation ──────────────────────────────────────────

# Patterns that indicate LLM self-annotation / inner-thought leakage
# in speech text.  These are stripped as a safety net even when the
# prompt already instructs the model to keep speech clean.
_THOUGHT_LEAK_PATTERNS: list[re.Pattern] = [
    # Full-width parentheses containing thought markers
    # e.g. "民（其实我是猎人但我不能说）" → "民"
    re.compile(
        r'（[^）]*(?:在心里|其实是|其实我是|备注|注意|我不能说|'
        r'我不能暴露|内心|心里|悄悄|偷偷|暗[中地]|实际[上是我]|'
        r'本身是|策略|战术|此处暂不|先不|不要暴露|避免暴露)[^）]*）'
    ),
    # Half-width parentheses containing thought markers
    re.compile(
        r'\([^)]*(?:其实是|其实我是|备注|心里|内心|'
        r'我不能说|悄悄|偷偷)[^)]*\)'
    ),
    # Trailing "（注：...）" or "（注意：...）" annotations
    re.compile(r'（注[：:][^）]*）'),
    re.compile(r'（注意[：:][^）]*）'),
]


def _sanitize_speech(speech: str) -> str:
    """Strip LLM self-annotation and inner-thought leakage from speech.

    As a safety net, removes parenthetical content that contains
    markers of internal reasoning (e.g. "在心里", "其实我是", "备注").
    This runs after prompt-based prevention to catch any remaining leaks.
    """
    for pat in _THOUGHT_LEAK_PATTERNS:
        speech = pat.sub("", speech)
    # Clean up double spaces / leading/trailing whitespace left by removals
    speech = re.sub(r' +', ' ', speech).strip()
    # Remove empty parentheses pairs that may remain
    speech = re.sub(r'（\s*）', '', speech)
    speech = re.sub(r'\(\s*\)', '', speech)
    return speech
