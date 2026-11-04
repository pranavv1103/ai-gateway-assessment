"""End-to-end checks over real sockets: every gateway talking to its mock backend."""

import httpx
import pytest

from aigw.config import GatewaySettings, LLMGatewaySettings, RouterSettings
from aigw.gateway.app import create_app as create_mcp_gateway
from aigw.gateway.mock_downstream import create_mock_downstream
from aigw.llm_gateway.app import create_app as create_llm_gateway
from aigw.llm_gateway.mock_provider import create_mock_llm
from aigw.router.app import create_app as create_router
from aigw.router.mock_provider import create_mock_provider
from helpers import serve


def test_mcp_gateway_end_to_end():
    downstream = create_mock_downstream()
    with serve(downstream) as downstream_url:
        settings = GatewaySettings(
            downstream_url=f"{downstream_url}/mcp",
            token_roles={"admin-token": "admin", "viewer-token": "viewer"},
        )
        with serve(create_mcp_gateway(settings)) as gateway_url:
            call = {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "admin_reset_key", "arguments": {}},
            }
            viewer = httpx.post(
                f"{gateway_url}/mcp", json=call, headers={"Authorization": "Bearer viewer-token"}
            )
            assert viewer.json()["error"]["code"] == -32001
            assert viewer.json()["id"] == 7
            admin = httpx.post(
                f"{gateway_url}/mcp", json=call, headers={"Authorization": "Bearer admin-token"}
            )
            assert "result" in admin.json()
            listing = httpx.post(
                f"{gateway_url}/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Authorization": "Bearer viewer-token"},
            )
            assert listing.status_code == 200


def test_llm_gateway_redacts_over_real_stream():
    with serve(create_mock_llm()) as provider_url:
        settings = LLMGatewaySettings(provider_url=f"{provider_url}/v1/chat/completions")
        with serve(create_llm_gateway(settings)) as gateway_url:
            body = {
                "model": "mock",
                "messages": [
                    {"role": "user", "content": "mail bob@example.com ssn 123-45-6789 ok"}
                ],
                "mock": {"delta_size": 3},  # tiny deltas split every value across frames
            }
            with httpx.stream("POST", f"{gateway_url}/v1/chat/completions", json=body) as r:
                text = "".join(r.iter_text())
    assert "bob@example.com" not in text
    assert "123-45-6789" not in text
    assert text.count("[REDACTED]") >= 2


@pytest.mark.parametrize("fault", [None, "429"])
def test_router_end_to_end(tmp_path, fault):
    with (
        serve(create_mock_provider("primary")) as primary,
        serve(create_mock_provider("secondary")) as secondary,
    ):
        settings = RouterSettings(
            primary_url=f"{primary}/v1/completions",
            secondary_url=f"{secondary}/v1/completions",
            db_path=tmp_path / "e2e.db",
        )
        with serve(create_router(settings)) as gateway:
            body = {"prompt": "hello"}
            if fault:
                body["mock"] = {"primary": {"status": 429}}
            response = httpx.post(
                f"{gateway}/v1/completions", json=body, headers={"Authorization": "Bearer k"}
            )
    assert response.status_code == 200
    assert response.json()["provider"] == ("secondary" if fault else "primary")
