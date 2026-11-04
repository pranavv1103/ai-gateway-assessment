import asyncio
import json
import time

import httpx
import pytest

from aigw.config import LLMGatewaySettings
from aigw.llm_gateway.app import create_app
from aigw.llm_gateway.mock_provider import create_mock_llm
from aigw.llm_gateway.sse import SSEParser
from helpers import serve

PROVIDER = "http://provider.internal.example:9002/v1/chat/completions"
REQUEST = {"model": "mock", "messages": [{"role": "user", "content": "hi"}]}


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None, delay=0.0):
        self.chunks, self.error, self.delay = chunks, error, delay
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
            if self.delay:
                await asyncio.sleep(self.delay)
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def sse_bytes(texts, finish=True, done=True) -> bytes:
    frames = []
    for text in texts:
        body = {"id": "c", "choices": [{"index": 0, "delta": {"content": text}}]}
        frames.append(f"data: {json.dumps(body, ensure_ascii=False)}\n\n")
    if finish:
        frames.append(
            'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
    if done:
        frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def split_bytes(raw: bytes, size: int) -> list[bytes]:
    return [raw[i : i + size] for i in range(0, len(raw), size)]


async def call_gateway(provider_handler, payload=None, settings=None):
    transport = httpx.MockTransport(provider_handler)
    app = create_app(
        settings or LLMGatewaySettings(provider_url=PROVIDER),
        httpx.AsyncClient(transport=transport),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        return await client.post(
            "/v1/chat/completions", json=REQUEST if payload is None else payload
        )


def response_text(response: httpx.Response) -> str:
    text = ""
    for event in SSEParser().feed(response.content):
        if event.data != "[DONE]":
            for choice in json.loads(event.data).get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
    return text


def sse_response(raw_chunks, **kwargs):
    stream = ChunkedStream(raw_chunks, **kwargs)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream), stream


# ---------- redaction through the full HTTP path ----------


@pytest.mark.parametrize("network_chunk", [1, 2, 3, 7, 13, 64, 10_000])
async def test_pii_redacted_regardless_of_network_chunking(network_chunk):
    texts = [
        "My email is john",
        ".doe@exam",
        "ple.com and SSN 123-4",
        "5-6789, card 4111 11",
        "11 1111 1111.",
    ]
    raw = sse_bytes(texts)

    def handler(request):
        response, _ = sse_response(split_bytes(raw, network_chunk))
        return response

    response = await call_gateway(handler)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response_text(response)
    assert text == "My email is [REDACTED] and SSN [REDACTED], card [REDACTED]."
    assert b"john" not in response.content and b"6789" not in response.content


async def test_normal_text_and_unicode_unchanged():
    texts = ["Héllo wörld ", "日本語のテキスト ", "emoji 😀 done."]

    def handler(request):
        return sse_response(split_bytes(sse_bytes(texts), 5))[0]

    assert response_text(await call_gateway(handler)) == "".join(texts)


async def test_multibyte_character_split_across_network_chunks():
    raw = sse_bytes(["price: 5€ — fine 😀 ", "next"])
    euro = raw.index("€".encode())
    emoji = raw.index("😀".encode())
    chunks = [raw[: euro + 1], raw[euro + 1 : emoji + 2], raw[emoji + 2 :]]

    def handler(request):
        return sse_response(chunks)[0]

    assert response_text(await call_gateway(handler)) == "price: 5€ — fine 😀 next"


async def test_all_three_pii_types_in_one_delta():
    texts = ["a@b.io 123-45-6789 4111111111111111 ok"]

    def handler(request):
        return sse_response([sse_bytes(texts)])[0]

    assert response_text(await call_gateway(handler)) == "[REDACTED] [REDACTED] [REDACTED] ok"


async def test_request_forwarded_with_streaming_forced_and_no_client_credentials():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return sse_response([sse_bytes(["x"])])[0]

    transport = httpx.MockTransport(handler)
    app = create_app(
        LLMGatewaySettings(provider_url=PROVIDER),
        httpx.AsyncClient(transport=transport),
        provider_api_key="server-side-key",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            json={**REQUEST, "stream": False},
            headers={"Authorization": "Bearer client-secret", "Cookie": "s=1"},
        )
    assert seen["body"]["stream"] is True
    assert seen["headers"]["authorization"] == "Bearer server-side-key"
    assert "cookie" not in seen["headers"]


async def test_malformed_frames_dropped_stream_continues():
    raw = b"data: {not json}\n\n" + sse_bytes(["good text "]) + b""

    def handler(request):
        return sse_response(split_bytes(raw, 9))[0]

    response = await call_gateway(handler)
    assert response_text(response) == "good text "
    assert b"not json" not in response.content


async def test_truncated_final_frame_is_discarded():
    raw = (
        sse_bytes(["hello "], finish=False, done=False)
        + b'data: {"choices":[{"index":0,"delta":{"con'
    )

    def handler(request):
        return sse_response([raw])[0]

    response = await call_gateway(handler)
    assert response_text(response) == "hello "


# ---------- upstream failures ----------


async def test_midstream_upstream_error_flushes_text_then_emits_sanitized_error_event():
    raw = sse_bytes(["partial answer for a@b.com tail"], finish=False, done=False)

    def handler(request):
        return sse_response([raw], error=httpx.ReadError("conn reset by 10.0.0.5"))[0]

    response = await call_gateway(handler)
    assert response.status_code == 200
    events = SSEParser().feed(response.content)
    assert events[-1].event == "error"
    error = json.loads(events[-1].data)["error"]
    assert error == {"type": "upstream_error", "message": "Upstream stream interrupted"}
    assert "10.0.0.5" not in response.text
    assert "partial answer for [REDACTED] tail" in response_text(response)


@pytest.mark.parametrize(
    ("upstream", "expected_status"),
    [
        (httpx.Response(500, text="Traceback ... /srv/provider/app.py"), 502),
        (httpx.Response(401, text="bad key sk-secret"), 502),
        (httpx.Response(429, text="slow down"), 429),
        (
            httpx.Response(200, text="<html>not sse</html>", headers={"content-type": "text/html"}),
            502,
        ),
    ],
)
async def test_upstream_http_errors_are_sanitized(upstream, expected_status):
    response = await call_gateway(lambda request: upstream)
    assert response.status_code == expected_status
    body = response.json()["error"]
    assert set(body) == {"type", "message"}
    for leaked in ("Traceback", "/srv", "sk-secret", "provider.internal", "html"):
        assert leaked not in response.text


async def test_upstream_connect_error_is_502():
    def handler(request):
        raise httpx.ConnectError("refused provider.internal.example:9002")

    response = await call_gateway(handler)
    assert response.status_code == 502
    assert "provider.internal" not in response.text


async def test_upstream_timeout_is_504():
    def handler(request):
        raise httpx.ConnectTimeout("timeout")

    response = await call_gateway(handler)
    assert response.status_code == 504
    assert response.json()["error"]["type"] == "upstream_timeout"


async def test_oversized_event_terminates_stream_with_error_event():
    huge = b"data: " + b"x" * 5000 + b"\n\n"
    settings = LLMGatewaySettings(provider_url=PROVIDER, max_event_chars=1000)

    def handler(request):
        return sse_response([huge])[0]

    response = await call_gateway(handler, settings=settings)
    events = SSEParser().feed(response.content)
    assert events[-1].event == "error"


@pytest.mark.parametrize(
    "payload",
    [{}, {"model": "m"}, {"model": "m", "messages": []}, {"model": "", "messages": [{}]}, []],
)
async def test_invalid_client_requests_rejected_without_upstream_call(payload):
    calls = []

    def handler(request):
        calls.append(request)
        return sse_response([sse_bytes(["x"])])[0]

    response = await call_gateway(handler, payload=payload)
    assert response.status_code == 422
    assert calls == []


async def test_upstream_response_closed_after_normal_stream():
    holder = {}

    def handler(request):
        response, stream = sse_response([sse_bytes(["hi"])])
        holder["stream"] = stream
        return response

    await call_gateway(handler)
    assert holder["stream"].closed


# ---------- large streams ----------


async def test_large_stream_is_fully_redacted():
    texts = [
        f"chunk {i} contact user{i}@example.com ssn 123-45-{1000 + i % 8000} " for i in range(3000)
    ]
    raw = sse_bytes(texts)

    def handler(request):
        return sse_response(split_bytes(raw, 1500))[0]

    response = await call_gateway(handler)
    text = response_text(response)
    assert text.count("[REDACTED]") == 6000
    assert "@example.com" not in text
    assert "123-45-" not in text


# ---------- live sockets: TTFT, cancellation, real upstream failure ----------


def test_live_stream_ttft_is_low_and_redaction_works():
    provider = create_mock_llm()
    message = "write to ceo@corp.example.com and quote ssn 123-45-6789 " * 4
    with serve(provider) as provider_url:
        settings = LLMGatewaySettings(provider_url=f"{provider_url}/v1/chat/completions")
        with serve(create_app(settings)) as gateway_url:
            body = {
                "model": "m",
                "messages": [{"role": "user", "content": message}],
                "mock": {"delta_size": 3, "delay_ms": 20},
            }
            start = time.perf_counter()
            first_data_at = None
            raw = b""
            with httpx.stream(
                "POST", f"{gateway_url}/v1/chat/completions", json=body, timeout=30
            ) as r:
                assert r.status_code == 200
                for chunk in r.iter_raw():
                    if first_data_at is None:
                        first_data_at = time.perf_counter() - start
                    raw += chunk
            total = time.perf_counter() - start
    text = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for event in SSEParser().feed(raw)
        if event.data != "[DONE]"
        for choice in json.loads(event.data)["choices"]
    )
    assert "ceo@corp" not in text and "123-45-6789" not in text
    assert text.count("[REDACTED]") == 8
    assert first_data_at < 0.5
    assert first_data_at < total / 4  # first token arrives long before the stream completes


def test_client_disconnect_closes_upstream_stream():
    provider = create_mock_llm()
    with serve(provider) as provider_url:
        settings = LLMGatewaySettings(provider_url=f"{provider_url}/v1/chat/completions")
        with serve(create_app(settings)) as gateway_url:
            body = {
                "model": "m",
                "messages": [{"role": "user", "content": "word " * 500}],
                "mock": {"delta_size": 5, "delay_ms": 20},
            }
            with httpx.stream(
                "POST", f"{gateway_url}/v1/chat/completions", json=body, timeout=30
            ) as r:
                next(r.iter_raw())
                deadline = time.monotonic() + 3
                while provider.state.open_streams != 1 and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert provider.state.open_streams == 1
            # client closed the connection mid-stream; the gateway must drop the provider stream
            deadline = time.monotonic() + 5
            while provider.state.open_streams != 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert provider.state.open_streams == 0
            assert provider.state.closed_streams == 1


def test_live_upstream_failure_mid_stream_yields_error_event():
    provider = create_mock_llm()
    with serve(provider) as provider_url:
        settings = LLMGatewaySettings(provider_url=f"{provider_url}/v1/chat/completions")
        with serve(create_app(settings)) as gateway_url:
            body = {
                "model": "m",
                "messages": [{"role": "user", "content": "hello there general kenobi"}],
                "mock": {"delta_size": 4, "fail_after": 3},
            }
            response = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30)
    events = SSEParser().feed(response.content)
    assert events[-1].event == "error"
    assert "Traceback" not in response.text and "RuntimeError" not in response.text


def test_live_mock_provider_error_status_is_sanitized():
    provider = create_mock_llm()
    with serve(provider) as provider_url:
        settings = LLMGatewaySettings(provider_url=f"{provider_url}/v1/chat/completions")
        with serve(create_app(settings)) as gateway_url:
            body = {**REQUEST, "mock": {"status": 503}}
            response = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30)
    assert response.status_code == 502
    assert "mock failure" not in response.text
