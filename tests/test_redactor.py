import random
import tracemalloc

import pytest

from aigw.llm_gateway.redactor import (
    MAX_MATCH_LENGTH,
    REDACTED,
    StreamRedactor,
    luhn_valid,
    redact_text,
)


def stream(chunks, **kwargs) -> str:
    redactor = StreamRedactor(**kwargs)
    out = [redactor.feed(chunk) for chunk in chunks]
    out.append(redactor.flush())
    return "".join(out)


def char_chunks(text: str, size: int = 1):
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---------- complete values ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail john.doe@example.com now", f"mail {REDACTED} now"),
        ("a+tag@sub.mail.example.co.uk.", f"{REDACTED}."),
        ("UPPER@EXAMPLE.ORG", REDACTED),
        ("ssn 123-45-6789 ok", f"ssn {REDACTED} ok"),
        ("ssn 123 45 6789 ok", f"ssn {REDACTED} ok"),
        ("ssn 987-65-4321.", f"ssn {REDACTED}."),
        ("card 4111 1111 1111 1111 ok", f"card {REDACTED} ok"),
        ("card 4111-1111-1111-1111", f"card {REDACTED}"),
        ("card 4111111111111111", f"card {REDACTED}"),
        ("card 1234-5678-9012-3456", f"card {REDACTED}"),  # grouped: format alone is enough
        ("amex 3782 822463 10005", f"amex {REDACTED}"),
        ("amex 378282246310005", f"amex {REDACTED}"),
        ("a@b.com and c@d.org", f"{REDACTED} and {REDACTED}"),
        ("both 123-45-6789 and x@y.io", f"both {REDACTED} and {REDACTED}"),
    ],
)
def test_complete_values_are_redacted(text, expected):
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "plain text with no pii",
        "version 1.2.3 released 2024-01-15",
        "order 12345 and 1234567890123456789012",  # 22 digits, not a card length
        "timestamp 1699999999999",  # 13 digits but Luhn-invalid
        "phone 555-1234",
        "not an email: user@localhost",
        "at sign @ alone and user@ and @domain.com",
        "1234 5678 9012",  # only 12 digits
        "ssn-like 12-345-6789",
        "x" * 1000,
    ],
)
def test_non_pii_is_untouched(text):
    assert redact_text(text) == text


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_digits_adjacent_to_longer_numbers_are_not_ssn():
    assert redact_text("9123-45-67890") == "9123-45-67890"


# ---------- fragmentation ----------


FRAGMENT_CASES = [
    "Contact john.doe@example.com for details.",
    "My SSN is 123-45-6789, thanks.",
    "Card: 4111 1111 1111 1111 expires soon.",
    "Card: 4111-1111-1111-1111.",
    "plain 4111111111111111 digits",
    "x@y.co then a@b.org then 111-22-3333 then 4242 4242 4242 4242 end",
    "unicode café résumé 日本語 emoji 😀 mail me@ex.com 😀",
    "no pii here, just words and punctuation!",
    "ends with partial email john@exa",
    "ends with digits 123-45",
]


@pytest.mark.parametrize("text", FRAGMENT_CASES)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16])
def test_fixed_size_fragments_match_whole_text_redaction(text, size):
    assert stream(char_chunks(text, size)) == redact_text(text)


@pytest.mark.parametrize("text", FRAGMENT_CASES)
def test_random_fragmentation_matches_whole_text_redaction(text):
    rng = random.Random(1234)
    for _ in range(200):
        chunks, i = [], 0
        while i < len(text):
            step = rng.randint(1, 9)
            chunks.append(text[i : i + step])
            i += step
        assert stream(chunks) == redact_text(text)


def test_property_random_documents():
    rng = random.Random(7)
    parts = [
        "hello",
        " ",
        "world",
        ".",
        ",",
        "\n",
        "john@example.com",
        "a.b@c.dev",
        "123-45-6789",
        "4111 1111 1111 1111",
        "4111111111111111",
        "12",
        "345",
        "-",
        "@",
        "x",
        "99",
        " 4",
        "5 ",
        "mail:",
        "(",
        ")",
        "é",
        "[",
        "]",
    ]
    for _ in range(300):
        text = "".join(rng.choice(parts) for _ in range(rng.randint(1, 40)))
        chunks, i = [], 0
        while i < len(text):
            step = rng.randint(1, 12)
            chunks.append(text[i : i + step])
            i += step
        assert stream(chunks) == redact_text(text), (text, chunks)


@pytest.mark.parametrize("split_at", range(1, len("john.doe@example.com")))
def test_email_split_at_every_position_never_leaks_a_fragment(split_at):
    email = "john.doe@example.com"
    out = stream([f"say {email[:split_at]}", f"{email[split_at:]} bye"])
    assert out == f"say {REDACTED} bye"


@pytest.mark.parametrize("split_at", range(1, len("123-45-6789")))
def test_ssn_split_at_every_position(split_at):
    ssn = "123-45-6789"
    assert stream([f"id {ssn[:split_at]}", f"{ssn[split_at:]}."]) == f"id {REDACTED}."


@pytest.mark.parametrize("split_at", range(1, len("4111 1111 1111 1111")))
def test_card_split_at_every_position(split_at):
    card = "4111 1111 1111 1111"
    assert stream([f"pay {card[:split_at]}", f"{card[split_at:]} now"]) == f"pay {REDACTED} now"


def test_pii_split_across_many_single_character_deltas():
    text = "reach me at jane_roe+x@mail.example.com or 078-05-1120 or 5500 0000 0000 0004!"
    assert stream(char_chunks(text)) == (f"reach me at {REDACTED} or {REDACTED} or {REDACTED}!")


# ---------- latency / holdback ----------


def test_plain_words_are_released_promptly():
    redactor = StreamRedactor()
    assert redactor.feed("Hello ") == "Hello "
    assert redactor.feed("world") == ""  # trailing word held in case it becomes an email
    assert redactor.feed("! ") == "world! "
    assert redactor.pending_length == 0


def test_holdback_is_released_on_flush():
    redactor = StreamRedactor()
    assert redactor.feed("trailing") == ""
    assert redactor.flush() == "trailing"
    assert redactor.flush() == ""


def test_empty_feed_is_noop():
    assert StreamRedactor().feed("") == ""


def test_incomplete_pii_at_end_of_stream_is_not_redacted():
    # Documented limitation: an unfinished value is indistinguishable from ordinary text.
    assert stream(["call 123-45-"]) == "call 123-45-"


def test_max_pending_rejects_too_small_bound():
    with pytest.raises(ValueError):
        StreamRedactor(max_pending=10)


# ---------- bounded memory ----------


def test_pending_buffer_stays_bounded_on_pathological_input():
    redactor = StreamRedactor()
    longest = 0
    for _ in range(2000):
        redactor.feed("a" * 100)
        longest = max(longest, redactor.pending_length)
    assert longest <= MAX_MATCH_LENGTH + 100


def test_long_token_with_pii_at_the_end_is_still_redacted():
    text = "b" * 5000 + " " + "c" * 600 + "x@y.com"
    expected = redact_text(text)
    assert REDACTED in expected
    assert stream(char_chunks(text, 37)) == expected


def test_large_stream_memory_is_bounded_and_output_correct():
    unit = "The quick brown fox mailed a@b.com with 123-45-6789 and some filler text. "
    repeats = 40_000  # ~3 MB of text
    redactor = StreamRedactor()
    tracemalloc.start()
    emitted = 0
    redactions = 0
    for _ in range(repeats):
        out = redactor.feed(unit)
        emitted += len(out)
        redactions += out.count(REDACTED)
    out = redactor.flush()
    emitted += len(out)
    redactions += out.count(REDACTED)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert redactions == 2 * repeats
    assert peak < 200_000  # far below the ~3 MB of text processed: nothing accumulates
    assert emitted == len(redact_text(unit)) * repeats


def test_full_text_never_contains_raw_pii_after_streaming():
    text = ("Reach a@b.com or 123-45-6789 or 4111 1111 1111 1111. " * 50) + "end"
    out = stream(char_chunks(text, 3))
    for secret in ("a@b.com", "123-45-6789", "4111"):
        assert secret not in out
