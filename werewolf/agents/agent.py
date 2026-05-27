"""LLM-based game agent with personality, role, and multi-turn memory."""

import json
import logging
from werewolf.agents.prompt_builder import build_system_prompt, build_user_message
from werewolf.agents.decision_parser import parse_llm_response, AgentOutput
from werewolf.llm import LLMClient

logger = logging.getLogger(__name__)

# Max conversation turns to keep in history (per agent)
MAX_HISTORY_TURNS = 20


class Agent:
    """An LLM-powered Werewolf player agent.

    Maintains multi-turn conversation history across decide() calls.
    Per both Qwen and DeepSeek docs, reasoning_content is NOT included
    in follow-up context — only assistant content is kept.
    """

    def __init__(self, player_id: int, role: str, personality: dict,
                 llm_client: LLMClient):
        self.player_id = player_id
        self.role = role
        self.personality = personality
        self.llm_client = llm_client
        self.is_alive = True

        # Private info (e.g., seer check results)
        self._private_info: list[str] = []

        # Multi-turn conversation history (system prompt + all turns)
        self._system_prompt = build_system_prompt(player_id, role, personality)
        self._history: list[dict] = [
            {"role": "system", "content": self._system_prompt}
        ]

    def add_private_info(self, info: str) -> None:
        """Add private info the agent knows (e.g., seer check result)."""
        self._private_info.append(info)

    async def decide(self, context: dict) -> AgentOutput | None:
        """Make a decision based on the current game context.

        Maintains conversation history across calls. Each call appends
        the user message and the assistant's response (content only,
        NOT reasoning_content) to the history.

        Returns None if the LLM call fails.
        """
        # Build the current user message
        user_msg = build_user_message(context)

        # Prepend private info
        if self._private_info:
            user_msg = "【你已知的信息】\n" + "\n".join(
                f"- {info}" for info in self._private_info
            ) + "\n\n" + user_msg

        # Append user message to history
        self._history.append({"role": "user", "content": user_msg})

        # Build messages for this call (system + recent history)
        messages = self._build_messages_for_call()

        # Use json_object format to encourage structured output
        response_format = {"type": "json_object"}

        try:
            response = await self.llm_client.chat(
                messages, response_format=response_format)
        except Exception as e:
            logger.error(
                "LLM call failed for player %d (phase: %s): %s",
                self.player_id, context.get("phase", "?"), e)
            # Remove the user message we just added (failed call)
            self._history.pop()
            return AgentOutput(
                speech=f"（{self.player_id}号玩家暂时无法发言）")

        output = parse_llm_response(response)

        # Extract content only (NOT reasoning_content) for history,
        # per both Qwen and DeepSeek multi-turn docs
        assistant_content = self._extract_assistant_content(response, output)

        # Append assistant response to history (content only, no reasoning)
        self._history.append({
            "role": "assistant",
            "content": assistant_content,
        })

        # Trim history if too long (keep system prompt + recent turns)
        self._trim_history()

        logger.debug("Player %d (%s) phase=%s speech=%s... action=%s",
                      self.player_id, self.role,
                      context.get("phase", "?"),
                      (output.speech or "")[:80],
                      output.action)
        return output

    def _build_messages_for_call(self) -> list[dict]:
        """Build the messages array for the current API call.

        Strategy: system prompt + truncated history + current user message.
        The current user message is already in self._history as the last entry.
        """
        # System prompt is always first
        messages = [self._history[0]]

        # Add recent history, skipping system prompt
        history_turns = self._history[1:]
        messages.extend(history_turns)

        return messages

    def _extract_assistant_content(self, response: dict,
                                   output: AgentOutput) -> str:
        """Extract the content portion of the assistant response for history.

        Returns output JSON as a string for context preservation.
        reasoning_content is intentionally excluded per API best practice.
        """
        # Use the parsed output as a JSON string — cleaner than raw content
        content_obj = {
            "thought": output.thought,
            "speech": output.speech,
            "action": output.action,
        }
        return json.dumps(content_obj, ensure_ascii=False)

    def _trim_history(self) -> None:
        """Trim conversation history to prevent token overflow.

        Keeps: system prompt + last MAX_HISTORY_TURNS exchanges.
        """
        # Each "turn" = user + assistant = 2 messages
        max_messages = 1 + MAX_HISTORY_TURNS * 2  # system + N turns
        if len(self._history) > max_messages:
            # Keep system prompt (index 0) + recent messages
            excess = len(self._history) - max_messages
            self._history = [self._history[0]] + self._history[1 + excess:]

    def reset_history(self) -> None:
        """Reset conversation history (keep system prompt, clear turns)."""
        self._history = [
            {"role": "system", "content": self._system_prompt}
        ]


def create_agent_factory(llm_client: LLMClient):
    """Return a factory function for creating agents with a shared LLM client."""

    def factory(player_id: int, role: str, personality: dict) -> Agent:
        return Agent(
            player_id=player_id,
            role=role,
            personality=personality,
            llm_client=llm_client,
        )

    return factory
