import pytest
from pydantic import ValidationError

from aigw.mcp_server.models import GetCustomerRecordInput, TriggerRefundInput

VALID_REASON = "Customer received damaged goods"


def refund(**overrides):
    data = {"customer_id": "CUST-12345", "amount": 25.5, "reason": VALID_REASON}
    data.update(overrides)
    return TriggerRefundInput.model_validate(data)


@pytest.mark.parametrize("cid", ["CUST-12345", "CUST-00000", "CUST-99998"])
def test_valid_customer_ids(cid):
    assert GetCustomerRecordInput(customer_id=cid).customer_id == cid


@pytest.mark.parametrize(
    "cid",
    [
        "CUST-1234",
        "CUST-123456",
        "cust-12345",
        "CUST12345",
        "CUST-1234A",
        " CUST-12345",
        "CUST-12345 ",
        "CUST-12345\n",
        "CUST-١٢٣٤٥",
        "",
        "CUST-",
        "XCUST-12345",
    ],
)
def test_invalid_customer_ids(cid):
    with pytest.raises(ValidationError):
        GetCustomerRecordInput(customer_id=cid)


@pytest.mark.parametrize("bad", [None, 12345, ["CUST-12345"], {"a": 1}, True])
def test_customer_id_wrong_types(bad):
    with pytest.raises(ValidationError):
        GetCustomerRecordInput.model_validate({"customer_id": bad})


def test_extra_and_missing_fields_rejected():
    with pytest.raises(ValidationError):
        GetCustomerRecordInput.model_validate({"customer_id": "CUST-12345", "extra": 1})
    with pytest.raises(ValidationError):
        GetCustomerRecordInput.model_validate({})
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate({"customer_id": "CUST-12345", "amount": 1})


@pytest.mark.parametrize("amount", [0.01, 1, 10.5, 99999.99, 100000])
def test_valid_amounts(amount):
    assert refund(amount=amount).amount > 0


def test_float_amount_is_exact_decimal():
    assert str(refund(amount=0.1).amount) == "0.1"


@pytest.mark.parametrize(
    "amount",
    [
        0,
        0.0,
        -1,
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        "10",
        "NaN",
        True,
        False,
        None,
        [],
        {},
        1.005,
        100000.01,
        1e12,
    ],
)
def test_invalid_amounts(amount):
    with pytest.raises(ValidationError):
        refund(amount=amount)


@pytest.mark.parametrize(
    "reason",
    ["", "short", "         ", "123456789", "!!!!!!!!!!!!", "  tiny   ", "x" * 501, None, 5],
)
def test_invalid_reasons(reason):
    with pytest.raises(ValidationError):
        refund(reason=reason)


def test_reason_is_trimmed_and_boundary_accepted():
    assert refund(reason="  1234567890  ").reason == "1234567890"


def test_refund_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        TriggerRefundInput.model_validate(
            {"customer_id": "CUST-12345", "amount": 5, "reason": VALID_REASON, "admin": True}
        )
