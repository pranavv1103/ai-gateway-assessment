"""Applies redaction to OpenAI-style streaming chat/completion chunks."""

import json
import logging
from typing import Any

from aigw.llm_gateway.redactor import StreamRedactor
from aigw.llm_gateway.sse import SSEEvent, encode_event

logger = logging.getLogger(__name__)

MAX_CHOICES = 16
DONE_MARKER = "[DONE]"


class ChunkTransformer:
    """Per-response state: one redactor per choice index. Synchronous and easy to test."""

    def __init__(self) -> None:
        self._redactors: dict[int, StreamRedactor] = {}
        self._text_field: dict[int, str] = {}
        self._envelope: dict[str, Any] = {}
        self.malformed_events = 0
        self.done = False

    def _redactor(self, index: int) -> StreamRedactor | None:
        redactor = self._redactors.get(index)
        if redactor is None:
            if len(self._redactors) >= MAX_CHOICES:
                return None
            redactor = self._redactors[index] = StreamRedactor()
        return redactor

    def process_event(self, event: SSEEvent) -> list[bytes]:
        if self.done:
            return []
        data = event.data.strip()
        if data == DONE_MARKER:
            self.done = True
            return [*self._flush_all(), encode_event(DONE_MARKER)]
        try:
            chunk = json.loads(data)
        except ValueError:
            chunk = None
        if not isinstance(chunk, dict):
            # Unparsable frames are dropped: their content cannot be inspected safely.
            self.malformed_events += 1
            logger.warning("Dropped malformed upstream SSE event (%d chars)", len(data))
            return []

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return [encode_event(json.dumps(chunk, separators=(",", ":")), event.event, event.id)]

        self._envelope = {k: v for k, v in chunk.items() if k not in ("choices", "usage")}
        suppress = bool(choices)
        kept: list[Any] = []
        for choice in choices:
            if not isinstance(choice, dict):
                kept.append(choice)
                suppress = False
                continue
            index = choice.get("index") if isinstance(choice.get("index"), int) else 0
            redactor = self._redactor(index)
            if redactor is None:
                continue
            container, key = self._locate_text(choice)
            finished = choice.get("finish_reason") is not None
            if container is not None:
                self._text_field[index] = key
                safe = redactor.feed(container[key])
                if finished:
                    safe += redactor.flush()
                container[key] = safe
            elif finished:
                tail = redactor.flush()
                if tail:
                    choice.setdefault("delta", {})["content"] = tail
            if self._has_visible_content(choice, container, key):
                suppress = False
            kept.append(choice)
        if suppress:
            return []
        chunk["choices"] = kept
        return [encode_event(json.dumps(chunk, separators=(",", ":")), event.event, event.id)]

    @staticmethod
    def _locate_text(choice: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta, "content"
        if isinstance(choice.get("text"), str):
            return choice, "text"
        return None, ""

    @staticmethod
    def _has_visible_content(
        choice: dict[str, Any], container: dict[str, Any] | None, key: str
    ) -> bool:
        """False only for a content-only delta whose text is fully held back."""
        if container is None or choice.get("finish_reason") is not None:
            return True
        if container[key] != "":
            return True
        if container is choice:
            return any(k not in ("index", "text", "logprobs") for k in choice)
        return any(k != "content" for k in container) or any(
            k not in ("index", "delta", "finish_reason", "logprobs") for k in choice
        )

    def _flush_all(self) -> list[bytes]:
        frames = []
        for index, redactor in self._redactors.items():
            tail = redactor.flush()
            if not tail:
                continue
            if self._text_field.get(index) == "text":
                choice: dict[str, Any] = {"index": index, "text": tail, "finish_reason": None}
            else:
                choice = {"index": index, "delta": {"content": tail}, "finish_reason": None}
            body = {**self._envelope, "choices": [choice]}
            frames.append(encode_event(json.dumps(body, separators=(",", ":"))))
        return frames

    def finish(self) -> list[bytes]:
        """Upstream ended (or failed): release any held text."""
        if self.done:
            return []
        return self._flush_all()
