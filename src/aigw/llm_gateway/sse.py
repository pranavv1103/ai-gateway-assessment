"""Incremental Server-Sent Events parser/encoder (WHATWG EventSource framing rules)."""

import codecs
import re
from dataclasses import dataclass

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


class SSEProtocolError(Exception):
    """Raised when upstream framing exceeds safe bounds."""


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None


class SSEParser:
    """Feed raw bytes in arbitrary chunks; get complete events back.

    Handles UTF-8 sequences split across chunks, ``\\r\\n`` split across chunks, multi-line
    ``data:`` fields, comments, and enforces a bound on buffered (incomplete) data.
    Malformed UTF-8 is replaced with U+FFFD. An event still incomplete at EOF is discarded,
    as the SSE specification requires.
    """

    def __init__(self, max_buffered_chars: int = 1_048_576):
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._max_buffered = max_buffered_chars
        self._partial_line = ""
        self._skip_leading_lf = False
        self._at_stream_start = True
        self._data_lines: list[str] = []
        self._data_size = 0
        self._event: str | None = None
        self._id: str | None = None

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        text = self._decoder.decode(chunk)
        if not text:
            return []
        if self._at_stream_start:
            text = text.removeprefix("\ufeff")
            self._at_stream_start = False
        combined = self._partial_line + text
        if self._skip_leading_lf:
            combined = combined.removeprefix("\n")
            self._skip_leading_lf = False
        lines = _LINE_BREAK.split(combined)
        self._partial_line = lines.pop()
        if combined.endswith("\r"):
            self._skip_leading_lf = True
        if len(self._partial_line) > self._max_buffered:
            raise SSEProtocolError("SSE line exceeds maximum size")
        events = []
        for line in lines:
            event = self._handle_line(line)
            if event is not None:
                events.append(event)
        return events

    def _handle_line(self, line: str) -> SSEEvent | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            self._data_size += len(value) + 1
            if self._data_size > self._max_buffered:
                raise SSEProtocolError("SSE event exceeds maximum size")
            self._data_lines.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._id = value
        return None

    def _dispatch(self) -> SSEEvent | None:
        event = None
        if self._data_lines:
            event = SSEEvent("\n".join(self._data_lines), self._event, self._id)
        self._data_lines = []
        self._data_size = 0
        self._event = None
        return event


def encode_event(data: str, event: str | None = None, event_id: str | None = None) -> bytes:
    lines = []
    if event:
        lines.append(f"event: {event}")
    if event_id:
        lines.append(f"id: {event_id}")
    lines.extend(f"data: {line}" for line in data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode("utf-8")
