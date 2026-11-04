"""Pydantic models that define and enforce the MCP tool contracts."""

import math
import re
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MIN_REASON_LENGTH = 10
MAX_REASON_LENGTH = 500
MAX_REFUND_AMOUNT = Decimal("100000.00")

CustomerId = Annotated[str, Field(pattern=CUSTOMER_ID_PATTERN)]


def _coerce_amount(value: Any) -> Decimal:
    """Accept only real JSON numbers; never strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise ValueError("amount must be a number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return Decimal(repr(value))
    return Decimal(value)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)


class GetCustomerRecordInput(_StrictModel):
    customer_id: CustomerId


class TriggerRefundInput(_StrictModel):
    customer_id: CustomerId
    amount: Decimal
    reason: str = Field(max_length=MAX_REASON_LENGTH)

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, value: Any) -> Decimal:
        return _coerce_amount(value)

    @field_validator("amount")
    @classmethod
    def _check_amount(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value <= 0:
            raise ValueError("amount must be greater than zero")
        if value > MAX_REFUND_AMOUNT:
            raise ValueError(f"amount must not exceed {MAX_REFUND_AMOUNT}")
        if value != value.quantize(Decimal("0.01")):
            raise ValueError("amount must have at most 2 decimal places")
        return value

    @field_validator("reason")
    @classmethod
    def _check_reason(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < MIN_REASON_LENGTH:
            raise ValueError(f"reason must be at least {MIN_REASON_LENGTH} characters")
        if len(re.findall(r"\w", stripped)) < MIN_REASON_LENGTH // 2:
            raise ValueError("reason must contain meaningful text")
        return stripped


def tool_input_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema advertised to clients via tools/list."""
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    amount = schema.get("properties", {}).get("amount")
    if amount is not None:
        title = amount.get("title", "Amount")
        amount.clear()
        amount.update(
            title=title,
            type="number",
            exclusiveMinimum=0,
            maximum=float(MAX_REFUND_AMOUNT),
            description="Refund amount in major currency units, at most 2 decimal places",
        )
    return schema
