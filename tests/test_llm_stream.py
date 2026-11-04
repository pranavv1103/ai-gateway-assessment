import json

import pytest

from aigw.llm_gateway.sse import SSEParser, SSEProtocolError, encode_event
from aigw.llm_gateway.transform import ChunkTransformer


def parse_all(chunks) -> list:
    parser = SSEParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    return events


# ---------- SSE parser ----------


def test_basic_events():
    events = parse_all([b"data: one\n\ndata: two\n\n"])
    assert [e.data for e in events] == ["one", "two"]


def test_event_split_across_arbitrary_chunk_boundaries():
    raw = b"event: msg\nid: 7\ndata: hello\ndata: world\n\ndata: next\n\n"
    for size in (1, 2, 3, 5, 11):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        events = parse_all(chunks)
        assert [(e.event, e.id, e.data) for e in events] == [
            ("msg", "7", "hello\nworld"),
            (None, "7", "next"),  # last event id persists per the SSE spec
        ]


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n", b"\r"])
def test_all_line_ending_styles(terminator):
    raw = b"data: a" + terminator + terminator + b"data: b" + terminator + terminator
    for size in (1, 2, 100):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert [e.data for e in parse_all(chunks)] == ["a", "b"]


def test_crlf_split_between_chunks_is_one_line_break():
    events = parse_all([b"data: a\r", b"\n\r", b"\ndata: b\n\n"])
    assert [e.data for e in events] == ["a", "b"]


def test_multibyte_utf8_split_across_chunks():
    raw = "data: héllo 日本語 😀\n\n".encode()
    for size in (1, 2, 3):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert parse_all(chunks)[0].data == "héllo 日本語 😀"


def test_invalid_utf8_is_replaced_not_fatal():
    events = parse_all([b"data: bad \xff\xfe bytes\n\n"])
    assert "\ufffd" in events[0].data


def test_comments_and_unknown_fields_ignored_and_bom_stripped():
    events = parse_all([b"\xef\xbb\xbf: keepalive\nretry: 5\nfoo: bar\ndata:x\n\n"])
    assert [e.data for e in events] == ["x"]


def test_event_without_data_is_not_dispatched():
    assert parse_all([b"event: ping\n\n"]) == []


def test_incomplete_event_at_eof_is_discarded():
    assert parse_all([b"data: partial"]) == []


def test_oversized_line_raises():
    parser = SSEParser(max_buffered_chars=100)
    with pytest.raises(SSEProtocolError):
        parser.feed(b"data: " + b"x" * 200)


def test_oversized_event_across_lines_raises():
    parser = SSEParser(max_buffered_chars=100)
    with pytest.raises(SSEProtocolError):
        for _ in range(20):
            parser.feed(b"data: xxxxxxxxxx\n")


def test_encode_event_roundtrip_with_multiline_data():
    encoded = encode_event("a\nb", event="e", event_id="1")
    assert encoded == b"event: e\nid: 1\ndata: a\ndata: b\n\n"
    event = parse_all([encoded])[0]
    assert (event.event, event.id, event.data) == ("e", "1", "a\nb")


# ---------- chunk transformer ----------


def chunk(text=None, finish=None, **extra):
    delta = {} if text is None else {"content": text}
    body = {
        "id": "c1",
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    body.update(extra)
    return json.dumps(body)


def run_transformer(payloads: list[str]) -> list[bytes]:
    from aigw.llm_gateway.sse import SSEEvent

    transformer = ChunkTransformer()
    out: list[bytes] = []
    for payload in payloads:
        out.extend(transformer.process_event(SSEEvent(payload)))
    out.extend(transformer.finish())
    return out


def decode_frames(frames: list[bytes]):
    events = parse_all(frames)
    text = ""
    for event in events:
        if event.data == "[DONE]":
            continue
        for choice in json.loads(event.data).get("choices", []):
            text += (choice.get("delta") or {}).get("content") or choice.get("text") or ""
    return events, text


def test_transformer_redacts_across_deltas_and_keeps_envelope():
    frames = run_transformer(
        [
            chunk("Email me at john.d"),
            chunk("oe@exam"),
            chunk("ple.com please"),
            chunk(None, finish="stop"),
            "[DONE]",
        ]
    )
    events, text = decode_frames(frames)
    assert text == "Email me at [REDACTED] please"
    assert events[-1].data == "[DONE]"
    first = json.loads(events[0].data)
    assert first["id"] == "c1" and first["model"] == "m"


def test_held_back_text_is_flushed_with_finish_chunk():
    frames = run_transformer([chunk("last word"), chunk(None, finish="stop"), "[DONE]"])
    events, text = decode_frames(frames)
    assert text == "last word"
    finish_chunks = [e for e in events if e.data != "[DONE]" and "stop" in e.data]
    assert len(finish_chunks) == 1


def test_flush_before_done_without_finish_reason():
    frames = run_transformer([chunk("tail"), "[DONE]"])
    events, text = decode_frames(frames)
    assert text == "tail"
    assert events[-1].data == "[DONE]"
    assert events[-2].data != "[DONE]"


def test_eof_without_done_still_flushes():
    frames = run_transformer([chunk("partial word")])
    assert decode_frames(frames)[1] == "partial word"


def test_fully_held_content_only_chunks_are_suppressed():
    frames = run_transformer([chunk("abc"), chunk("def"), chunk("ghi ")])
    assert len(frames) == 1  # one emission once a boundary is reached
    assert decode_frames(frames)[1] == "abcdefghi "


def test_role_and_usage_chunks_pass_through():
    role = json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    usage = json.dumps({"choices": [], "usage": {"total_tokens": 5}})
    frames = run_transformer([role, usage])
    events = parse_all(frames)
    assert json.loads(events[0].data)["choices"][0]["delta"] == {"role": "assistant"}
    assert json.loads(events[1].data)["usage"] == {"total_tokens": 5}


@pytest.mark.parametrize("payload", ["not json", "{broken", "[1,2]", '"str"', "42", "null"])
def test_malformed_events_are_dropped_not_forwarded(payload):
    transformer = ChunkTransformer()
    from aigw.llm_gateway.sse import SSEEvent

    assert transformer.process_event(SSEEvent(payload)) == []
    assert transformer.malformed_events == 1


def test_stream_continues_after_malformed_event():
    frames = run_transformer(["garbage", chunk("ok "), "[DONE]"])
    assert decode_frames(frames)[1] == "ok "


def test_multiple_choices_are_isolated():
    def multi(a, b):
        return json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"content": a}, "finish_reason": None},
                    {"index": 1, "delta": {"content": b}, "finish_reason": None},
                ]
            }
        )

    frames = run_transformer([multi("x@y.", "123-45-"), multi("com ", "6789 "), "[DONE]"])
    texts = {0: "", 1: ""}
    for event in parse_all(frames):
        if event.data == "[DONE]":
            continue
        for choice in json.loads(event.data)["choices"]:
            texts[choice["index"]] += choice["delta"]["content"]
    assert texts == {0: "[REDACTED] ", 1: "[REDACTED] "}


def test_legacy_text_completion_field_is_redacted():
    parts = [
        json.dumps({"choices": [{"index": 0, "text": t, "finish_reason": None}]})
        for t in ("my mail a@", "b.com ", "bye")
    ]
    frames = run_transformer([*parts, "[DONE]"])
    texts = ""
    for event in parse_all(frames):
        if event.data != "[DONE]":
            texts += "".join(c.get("text", "") for c in json.loads(event.data)["choices"])
    assert texts == "my mail [REDACTED] bye"


def test_events_after_done_are_ignored():
    frames = run_transformer([chunk("a "), "[DONE]", chunk("late x@y.com ")])
    assert "late" not in decode_frames(frames)[1]


def test_too_many_choice_indexes_are_capped():
    payload = json.dumps(
        {"choices": [{"index": i, "delta": {"content": "x "}} for i in range(100)]}
    )
    frames = run_transformer([payload])
    choices = json.loads(parse_all(frames)[0].data)["choices"]
    assert len(choices) == 16
