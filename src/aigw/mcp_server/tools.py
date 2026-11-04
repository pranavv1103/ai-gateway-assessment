"""Deterministic mock business logic behind the MCP tools."""

import hashlib
from decimal import Decimal
from typing import Any

from aigw.mcp_server.models import GetCustomerRecordInput, TriggerRefundInput

_FIRST_NAMES = ["Ava", "Liam", "Noah", "Mia", "Ethan", "Zoe", "Lucas", "Iris", "Omar", "Priya"]
_LAST_NAMES = ["Nguyen", "Garcia", "Patel", "Kim", "Silva", "Cohen", "Okafor", "Larsen"]
_TIERS = ["bronze", "silver", "gold", "platinum"]

NOT_FOUND_CUSTOMER_ID = "CUST-99999"


class ToolExecutionError(Exception):
    """A well-formed call that failed in business logic (reported as isError, not JSON-RPC)."""


def _digits(customer_id: str) -> int:
    return int(customer_id.removeprefix("CUST-"))


def _lifetime_spend(customer_id: str) -> Decimal:
    return Decimal(_digits(customer_id) % 9000 + 500) + Decimal("0.50")


def get_customer_record(args: GetCustomerRecordInput) -> dict[str, Any]:
    if args.customer_id == NOT_FOUND_CUSTOMER_ID:
        raise ToolExecutionError(f"Customer {args.customer_id} was not found")
    n = _digits(args.customer_id)
    first, last = _FIRST_NAMES[n % len(_FIRST_NAMES)], _LAST_NAMES[n % len(_LAST_NAMES)]
    return {
        "customer_id": args.customer_id,
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "tier": _TIERS[n % len(_TIERS)],
        "lifetime_spend": str(_lifetime_spend(args.customer_id)),
        "currency": "USD",
        "status": "active",
    }


def trigger_refund(args: TriggerRefundInput) -> dict[str, Any]:
    if args.customer_id == NOT_FOUND_CUSTOMER_ID:
        raise ToolExecutionError(f"Customer {args.customer_id} was not found")
    if args.amount > _lifetime_spend(args.customer_id):
        raise ToolExecutionError("Refund amount exceeds the customer's lifetime spend")
    digest = hashlib.sha256(f"{args.customer_id}|{args.amount}|{args.reason}".encode()).hexdigest()
    return {
        "refund_id": f"RFD-{digest[:12].upper()}",
        "customer_id": args.customer_id,
        "amount": str(args.amount.quantize(Decimal("0.01"))),
        "currency": "USD",
        "status": "queued",
        "reason": args.reason,
    }
