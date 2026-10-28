"""Chat model for the Heabsy inference API.

Heabsy (https://heabsy.com/platform) serves open models through an
OpenAI-compatible API at https://api.heabsy.com/v1. This class is a thin
wrapper around ``langchain_openai.ChatOpenAI`` with the Heabsy base URL,
the ``HEABSY_API_KEY`` default and ``stream_usage=True`` (the gateway
reports token usage on streamed responses). Reasoning text returned by
Heabsy models in ``reasoning_content`` is kept in
``AIMessage.additional_kwargs["reasoning_content"]`` for non-streaming
calls; ``ChatOpenAI`` drops it by default.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

HEABSY_BASE_URL = "https://api.heabsy.com/v1"


class ChatHeabsy(ChatOpenAI):
    """ChatOpenAI pointed at the Heabsy API.

    Example:
        from langchain_heabsy import ChatHeabsy

        llm = ChatHeabsy(model="qwen38")  # key from HEABSY_API_KEY
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_heabsy", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"heabsy_api_key": "HEABSY_API_KEY"}

    @classmethod
    def _get_default_model(cls) -> str:
        return "qwen38"

    def __init__(self, *, model: str = "qwen38", heabsy_api_key: str | None = None,
                 base_url: str = HEABSY_BASE_URL, **kwargs: Any) -> None:
        kwargs.setdefault("stream_usage", True)
        super().__init__(model=model, api_key=heabsy_api_key, base_url=base_url, **kwargs)

    def _create_chat_result(self, response: Any, run_manager: Any = None) -> AIMessage:
        message = super()._create_chat_result(response, run_manager)
        try:
            reasoning = response["choices"][0]["message"].get("reasoning_content")
        except (KeyError, IndexError, AttributeError, TypeError):
            reasoning = None
        if reasoning:
            message.additional_kwargs["reasoning_content"] = reasoning
        return message
