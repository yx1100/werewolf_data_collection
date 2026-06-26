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
        self.is_garbled = False


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
        new_thought, new_speech = _unpack_nested_json(speech_val, data.get("thought", ""))
        if new_speech != speech_val:
            # Successfully unpacked
            data["thought"] = new_thought
            data["speech"] = new_speech
        else:
            # Unpacking failed — speech is garbled raw JSON
            speech_val = ""  # clear garbled speech
            logger.warning(
                "Garbled speech: could not unpack nested JSON: %s",
                speech_val[:200])

    # Defensive sanitise: strip LLM self-annotation from speech
    speech_val = data.get("speech", "")
    if isinstance(speech_val, str):
        speech_val = _sanitize_speech(speech_val)

    result = AgentOutput(
        thought=data.get("thought", ""),
        speech=speech_val,
        action=data.get("action", {}),
    )

    # Detect garbled speech after all parsing
    if _is_garbled_speech(result.speech):
        result.is_garbled = True
        logger.warning(
            "Garbled speech detected for player, is_garbled=True: %s",
            result.speech[:200])

    return result


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
# Shared thought-marker keywords used by both full-width and half-width
# parenthesis patterns below.  Adding a keyword here makes both patterns
# pick it up automatically.
_THOUGHT_MARKERS_FULL: str = (
    r'在心里|其实是|其实我是|备注|注意|提示|我不能说|我不能暴露|'
    r'内心|心里|悄悄|偷偷|OS|我在想|我心里|暗[中地]|实际[上是我]|'
    r'本身是|策略|战术|此处暂不|先不|不要暴露|避免暴露|'
    r'注[：:]|注意[：:]|提示[：:]'
)

_THOUGHT_MARKERS_HALF: str = (
    r'其实是|其实我是|备注|注意|提示|心里|内心|我不能说|悄悄|偷偷|'
    r'OS|我在想|我心里|注[:]|注意[:]|提示[:]'
)

_THOUGHT_LEAK_PATTERNS: list[re.Pattern] = [
    # Full-width parentheses containing thought markers
    # e.g. "民（其实我是猎人但我不能说）" → "民"
    re.compile(
        r'（[^）]*(?:' + _THOUGHT_MARKERS_FULL + r')[^）]*）'
    ),
    # Half-width parentheses containing thought markers
    re.compile(
        r'\([^)]*(?:' + _THOUGHT_MARKERS_HALF + r')[^)]*\)'
    ),
]

# Generic long parenthetical content patterns.
# Strips any parenthetical block >= 30 chars that doesn't look like a
# normal speech aside — catches reasoning leaks that miss specific markers.
_GENERIC_LEAK_PATTERNS: list[re.Pattern] = [
    # Full-width parentheses with >= 30 non-whitespace chars
    re.compile(r'（[^）]{30,}）'),
    # Half-width parentheses with >= 30 non-whitespace chars
    re.compile(r'\([^)]{30,}\)'),
]


def _sanitize_speech(speech: str) -> str:
    """Strip LLM self-annotation and inner-thought leakage from speech.

    As a safety net, removes parenthetical content that contains
    markers of internal reasoning (e.g. "在心里", "其实我是", "备注").
    This runs after prompt-based prevention to catch any remaining leaks.
    """
    for pat in _THOUGHT_LEAK_PATTERNS:
        speech = pat.sub("", speech)
    # Strip long parenthetical content (>= 30 non-whitespace CJK chars)
    # that doesn't match specific keywords but is still clearly reasoning
    for pat in _GENERIC_LEAK_PATTERNS:
        speech = pat.sub("", speech)
    # Clean up double spaces / leading/trailing whitespace left by removals
    speech = re.sub(r' +', ' ', speech).strip()
    # Remove empty parentheses pairs that may remain
    speech = re.sub(r'（\s*）', '', speech)
    speech = re.sub(r'\(\s*\)', '', speech)
    return speech


def _unpack_nested_json(speech_val: str, current_thought: str) -> tuple[str, str]:
    """Unpack nested JSON from a speech field that looks like a JSON object.

    The LLM sometimes outputs the entire JSON structure as the speech value
    (double-encoded JSON).  This function tries multiple strategies to
    extract the inner speech/thought content.

    Returns (updated_thought, inner_speech).
    If all strategies fail, returns (current_thought, original_speech_val).
    """
    stripped = speech_val.strip()

    # Strategy 1: try json.loads (handles properly formatted double-encoded JSON)
    try:
        nested = json.loads(stripped)
        if isinstance(nested, dict):
            inner_speech = nested.get("speech", "")
            if inner_speech:
                inner_thought = nested.get("thought", "")
                return (current_thought or inner_thought, inner_speech)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: regex field extraction (handles malformed inner JSON)
    result = _extract_fields(stripped)
    if result:
        inner_speech = result.get("speech", "")
        if inner_speech:
            inner_thought = result.get("thought", "")
            return (current_thought or inner_thought, inner_speech)

    # All strategies failed
    return (current_thought, speech_val)


def _is_garbled_speech(speech: str) -> bool:
    """Check if speech content appears garbled or malformed.

    Returns True if speech:
    - Starts with '{' (raw JSON content leaked into speech)
    - Contains excessive repetition (same 20+ char block repeated 4+ times)
    """
    if not speech:
        return False

    # Raw JSON in speech
    if speech.startswith("{") and "thought" in speech:
        return True

    # Check for excessive repetition in long speech
    if len(speech) >= 200:
        for chunk_size in (20, 30):
            chunks = [speech[i:i + chunk_size]
                      for i in range(0, len(speech), chunk_size)]
            if len(chunks) < 5:
                continue
            consecutive_dupes = 0
            for i in range(len(chunks) - 1):
                c1 = chunks[i].strip()
                c2 = chunks[i + 1].strip()
                if c1 and c2 and c1 == c2:
                    consecutive_dupes += 1
                    if consecutive_dupes >= 3:
                        return True
                else:
                    consecutive_dupes = 0

    return False
