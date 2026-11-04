"""Streaming PII redaction with bounded memory.

Correctness idea: text is only emitted up to a *safe cut*, a position that no PII match can
straddle. Every PII pattern is built from a small alphabet (ASCII letters/digits and ``._%+@-``,
plus a space *between digits* for grouped numbers). A character outside that alphabet therefore
terminates any possible match, so cutting right before or after one is always safe. Only the
trailing partial token is held back, which keeps latency to about one word and memory bounded.
If a token has no safe cut within ``MAX_MATCH_LENGTH`` characters, the oldest characters are
force-flushed; no pattern can be longer than that, so the forced cut is still safe.

With this scheme, ``feed()``-ing arbitrary fragments and then ``flush()`` yields exactly the same
output as ``redact_text()`` applied to the concatenated text.
"""

import re

REDACTED = "[REDACTED]"

# Upper bound on any match: 64 (local) + 1 + 5*64 (labels) + 63 (TLD) = 448, plus lookaround slack.
MAX_MATCH_LENGTH = 450

_EMAIL = r"[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,5}[A-Za-z]{2,63}"
_SSN = r"(?<!\d)\d{3}(?P<ssn_sep>[- ])\d{2}(?P=ssn_sep)\d{4}(?!\d)"
_CARD_GROUPED = (
    r"(?<!\d)(?:\d{4}(?P<card_sep>[ -])\d{4}(?P=card_sep)\d{4}(?P=card_sep)\d{1,7}"
    r"|\d{4}(?P<amex_sep>[ -])\d{6}(?P=amex_sep)\d{5})(?!\d)"
)
_CARD_PLAIN = r"(?<!\d)\d{13,19}(?!\d)"

_PATTERN = re.compile(
    rf"(?P<email>{_EMAIL})|(?P<ssn>{_SSN})|(?P<card_grouped>{_CARD_GROUPED})"
    rf"|(?P<card_plain>{_CARD_PLAIN})",
    re.ASCII,
)
_QUICK_CHECK = re.compile(r"[@0-9]", re.ASCII)

_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+@-")
_DIGITS = frozenset("0123456789")


def luhn_valid(number: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(number)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _replace(match: re.Match[str]) -> str:
    # Contiguous digit runs need a valid Luhn checksum (avoids redacting timestamps/IDs);
    # grouped 4-4-4-4 style numbers are redacted on format alone.
    if match.lastgroup == "card_plain" and not luhn_valid(match.group()):
        return match.group()
    return REDACTED


def redact_text(text: str) -> str:
    """Redact emails, SSNs and credit card numbers in a complete piece of text."""
    if not _QUICK_CHECK.search(text):
        return text
    return _PATTERN.sub(_replace, text)


def _is_boundary_char(buf: str, index: int, length: int) -> bool:
    """True if buf[index] cannot be part of any PII match."""
    char = buf[index]
    if char in _ALPHABET:
        return False
    if char == " ":
        between_digits = (
            index > 0
            and buf[index - 1] in _DIGITS
            and (index + 1 >= length or buf[index + 1] in _DIGITS)
        )
        return not between_digits
    return True


class StreamRedactor:
    """Incremental redactor for one text stream."""

    def __init__(self, max_pending: int = MAX_MATCH_LENGTH):
        if max_pending < MAX_MATCH_LENGTH:
            raise ValueError("max_pending must be at least MAX_MATCH_LENGTH")
        self._max_pending = max_pending
        self._pending = ""

    @property
    def pending_length(self) -> int:
        return len(self._pending)

    def _safe_cut(self, buf: str) -> int:
        length = len(buf)
        lowest = max(1, length - self._max_pending + 1)
        for cut in range(length, lowest - 1, -1):
            if _is_boundary_char(buf, cut - 1, length) or (
                cut < length and _is_boundary_char(buf, cut, length)
            ):
                return cut
        return max(0, length - self._max_pending)

    def feed(self, text: str) -> str:
        """Add text; return the portion that is safe to emit now (already redacted)."""
        if not text:
            return ""
        buf = self._pending + text
        cut = self._safe_cut(buf)
        self._pending = buf[cut:]
        return redact_text(buf[:cut])

    def flush(self) -> str:
        """End of stream: redact and release whatever is still held."""
        tail, self._pending = self._pending, ""
        return redact_text(tail)
