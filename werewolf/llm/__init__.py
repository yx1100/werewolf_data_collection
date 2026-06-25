"""LLM client abstraction and implementations with deep thinking support."""

from abc import ABC, abstractmethod
from openai import AsyncOpenAI

# Valid model names per provider.
# Must stay in sync with the frontend dropdown in werewolf/web/templates/index.html.
VALID_MODELS: dict[str, list[str]] = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "qwen": ["qwen3.6-flash", "qwen3.7-plus", "qwen3.7-max"],
    "mimo": ["mimo-v2.5-pro", "mimo-v2.5"],
}


class LLMClient(ABC):
    """Abstract LLM client interface (OpenAI-compatible)."""

    def __init__(self, config: dict, deep_thinking: bool = False,
                 thinking_budget: int = 0, preserve_thinking: bool = False):
        self.config = config
        self.deep_thinking = deep_thinking
        self.thinking_budget = thinking_budget
        self.preserve_thinking = preserve_thinking

    @abstractmethod
    def get_provider_name(self) -> str:
        ...

    async def chat(self, messages: list[dict],
                   response_format: dict | None = None) -> dict:
        """Send chat completion request. Returns the API response dict.

        When deep_thinking is enabled, the response will include
        reasoning_content alongside content. Only content is kept in
        multi-turn history per API best practice.
        """
        kwargs = {
            "model": self.config.get("model", "default"),
            "messages": messages,
            self._completion_tokens_key(): self.config.get("max_tokens", 2048),
        }

        # Temperature is not supported in deep thinking mode for
        # DeepSeek and MiMo (per their respective API docs).
        if not (self.deep_thinking and self.get_provider_name() in ("deepseek", "mimo")):
            kwargs["temperature"] = self.config.get("temperature", 1.0)

        if response_format:
            kwargs["response_format"] = response_format

        # Build extra_body with provider-specific thinking params
        extra_body = self._build_thinking_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        # DeepSeek reasoning_effort is a top-level param
        thinking_top_level = self._build_thinking_top_level()
        kwargs.update(thinking_top_level)

        response = await self._client().chat.completions.create(**kwargs)
        return response.model_dump()

    def _build_thinking_extra_body(self) -> dict:
        """Build extra_body dict for thinking params. Override per provider."""
        return {}

    def _build_thinking_top_level(self) -> dict:
        """Build top-level thinking params. Override per provider."""
        return {}

    def _completion_tokens_key(self) -> str:
        """Parameter name for max completion tokens.

        Most OpenAI-compatible APIs accept ``max_tokens``, but some
        (e.g. MiMo) require the newer ``max_completion_tokens``.
        """
        return "max_tokens"

    @abstractmethod
    def _client(self) -> AsyncOpenAI:
        ...


class QwenClient(LLMClient):
    """Qwen (通义千问) API client via OpenAI-compatible endpoint.

    Deep thinking: enable_thinking=True passed via extra_body (NOT a standard
    OpenAI param, must use extra_body per Qwen docs).

    Models supporting thinking: qwen-plus, qwen-max, qwen-turbo, qwen-flash
    (qwen3 series, mixed thinking mode).
    """

    def __init__(self, config: dict, deep_thinking: bool = False,
                 thinking_budget: int = 0, preserve_thinking: bool = False):
        super().__init__(config, deep_thinking, thinking_budget, preserve_thinking)
        self._async_client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

    def get_provider_name(self) -> str:
        return "qwen"

    def _client(self) -> AsyncOpenAI:
        return self._async_client

    def _build_thinking_extra_body(self) -> dict:
        """Qwen: enable_thinking must go through extra_body."""
        if not self.deep_thinking:
            return {}
        body = {"enable_thinking": True}
        if self.thinking_budget > 0:
            body["thinking_budget"] = self.thinking_budget
        if self.preserve_thinking:
            body["preserve_thinking"] = True
        return body


class DeepSeekClient(LLMClient):
    """DeepSeek API client via OpenAI-compatible endpoint.

    Deep thinking: ``thinking: {type}`` via extra_body + ``reasoning_effort``
    as a top-level parameter.  Per the DeepSeek thinking-mode guide, the
    OpenAI SDK natively supports ``reasoning_effort`` at the top level, while
    ``thinking`` (a non-OpenAI-standard param) must go through ``extra_body``.

    Models: deepseek-v4-pro, deepseek-v4-flash (mixed thinking mode).
    Default thinking is ON for deepseek-v4-pro.
    """

    def __init__(self, config: dict, deep_thinking: bool = False,
                 thinking_budget: int = 0, preserve_thinking: bool = False):
        super().__init__(config, deep_thinking, thinking_budget, preserve_thinking)
        self._async_client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.deepseek.com"),
        )

    def get_provider_name(self) -> str:
        return "deepseek"

    def _client(self) -> AsyncOpenAI:
        return self._async_client

    def _build_thinking_extra_body(self) -> dict:
        """DeepSeek: thinking type control via extra_body.

        ``thinking`` is not an OpenAI-standard param — must use extra_body.
        """
        if self.deep_thinking:
            return {"thinking": {"type": "enabled"}}
        else:
            return {"thinking": {"type": "disabled"}}

    def _build_thinking_top_level(self) -> dict:
        """DeepSeek: reasoning_effort as a top-level OpenAI-native param."""
        if self.deep_thinking:
            return {"reasoning_effort": self.config.get(
                "reasoning_effort", "high")}
        return {}


class MiMoClient(LLMClient):
    """MiMo (小米) API client via OpenAI-compatible endpoint.

    MiMo uses the ``api-key`` HTTP header for authentication (not the
    standard Bearer token).  We pass it through ``default_headers`` so it
    is sent alongside the Bearer token that the OpenAI SDK emits
    automatically — both are accepted by the MiMo gateway.

    Deep thinking: controlled via ``thinking: {type: "enabled"/"disabled"}``
    in ``extra_body``.  This is a MiMo-specific extension, NOT a standard
    OpenAI parameter.

    Default thinking is ON for mimo-v2.5-pro and mimo-v2.5, so when
    deep_thinking is False we explicitly send ``"disabled"`` to turn it off.
    Temperature / top_p are NOT supported when thinking is enabled.
    """

    def __init__(self, config: dict, deep_thinking: bool = False,
                 thinking_budget: int = 0, preserve_thinking: bool = False):
        super().__init__(config, deep_thinking, thinking_budget, preserve_thinking)
        self._async_client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url",
                                "https://api.xiaomimimo.com/v1"),
            default_headers={"api-key": config["api_key"]},
        )

    def get_provider_name(self) -> str:
        return "mimo"

    def _client(self) -> AsyncOpenAI:
        return self._async_client

    def _completion_tokens_key(self) -> str:
        """MiMo uses max_completion_tokens (not max_tokens)."""
        return "max_completion_tokens"

    def _build_thinking_extra_body(self) -> dict:
        """MiMo: thinking type control via extra_body.

        MiMo v2.5-pro / v2.5 default to thinking ON, so we always send
        the parameter explicitly to match the requested state.
        """
        if self.deep_thinking:
            return {"thinking": {"type": "enabled"}}
        else:
            return {"thinking": {"type": "disabled"}}


def create_llm_client(provider: str, config: dict,
                      deep_thinking: bool = False,
                      thinking_budget: int = 0,
                      preserve_thinking: bool = False) -> LLMClient:
    """Factory: create an LLM client by provider name.

    Args:
        provider: "qwen", "deepseek", or "mimo"
        config: provider-specific config dict (api_key, model, base_url, etc.)
        deep_thinking: enable chain-of-thought reasoning before response
        thinking_budget: max reasoning tokens (0 = unlimited, Qwen only)
        preserve_thinking: keep reasoning in multi-turn context (Qwen only)
    """
    providers = {
        "qwen": QwenClient,
        "deepseek": DeepSeekClient,
        "mimo": MiMoClient,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(providers.keys())}")

    return providers[provider](
        config,
        deep_thinking=deep_thinking,
        thinking_budget=thinking_budget,
        preserve_thinking=preserve_thinking,
    )
