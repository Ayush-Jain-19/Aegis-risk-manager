"""
src/api/schemas.py

WHY THIS FILE EXISTS
---------------------
Pydantic schemas are the FIRST line of defense in this system, not a
formality. Every field constraint here is a deliberate gate: malformed or
adversarial input (a merchant integration bug, a negative amount, a bogus
timestamp) should fail LOUDLY and INSTANTLY at the schema boundary, before
it ever reaches feature engineering or the model. This matters doubly in
fraud detection — a crafted payload designed to slip past validation and
confuse the model is itself an attack surface, so "strict by default" is
the right posture, not an inconvenience to relax later.

DESIGN NOTE ON WHAT'S IN THE PAYLOAD (AND WHAT ISN'T)
--------------------------------------------------------
The task spec fixes this payload to what a merchant checkout SDK can
realistically know at authorization time: amount, merchant identity,
category, both parties' coordinates, the cardholder's date of birth, and
the transaction timestamp. It deliberately does NOT include job, gender,
city, or state — a checkout flow doesn't have a KYC profile handy. main.py
documents exactly how those missing training-time features are backfilled
at inference time (via the same encoder's "unseen category" fallback used
for genuinely unseen merchants), so that gap is handled explicitly rather
than silently.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TransactionInput(BaseModel):
    """Raw incoming JSON from a merchant checkout integration."""

    amount: float = Field(
        ..., gt=0,
        description="Transaction amount in the merchant's settlement currency. Must be > 0.",
    )
    merchant: str = Field(
        ..., min_length=1, max_length=200,
        description="Merchant name/identifier as sent by the checkout integration.",
    )
    category: str = Field(
        ..., min_length=1, max_length=100,
        description="Merchant category, e.g. 'grocery_pos', 'shopping_net'.",
    )
    merchant_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude.")
    merchant_long: float = Field(..., ge=-180, le=180, description="Merchant longitude.")
    customer_lat: float = Field(..., ge=-90, le=90, description="Cardholder's home latitude.")
    customer_long: float = Field(..., ge=-180, le=180, description="Cardholder's home longitude.")
    dob: date = Field(..., description="Cardholder date of birth, ISO-8601 (YYYY-MM-DD).")
    trans_time: datetime = Field(
        ..., description="Transaction timestamp, ISO-8601. Timezone-naive values are treated as UTC.",
    )

    # ------------------------------------------------------------------ #
    # GATING VALIDATORS
    # Each of these is a hard fail — FastAPI/Pydantic will raise a
    # RequestValidationError, caught by the custom handler in
    # exception_handlers.py, which returns HTTP 422 with a structured body
    # AND a safe fallback decision (see that file for why "REVIEW" is the
    # default fallback, never "APPROVE").
    # ------------------------------------------------------------------ #

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        # Belt-and-suspenders: Field(gt=0) above already enforces this, but
        # an explicit validator lets us raise a message a fraud analyst
        # would actually understand ("non-positive amounts are not valid
        # transactions") rather than pydantic's generic "greater than 0"
        # framing, and gives us a single place to extend this check (e.g.
        # adding a max-amount sanity ceiling) without touching Field(...).
        if v <= 0:
            raise ValueError("amount must be a positive number — a merchant cannot charge <= 0.")
        return v

    @field_validator("merchant", "category")
    @classmethod
    def strip_and_reject_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank or whitespace-only.")
        return v

    @field_validator("dob")
    @classmethod
    def dob_must_be_plausible(cls, v: date) -> date:
        # A DOB in the future, or implying an implausible age, is a strong
        # signal of either a broken integration or a deliberately malformed
        # payload — either way we want a hard validation failure here
        # rather than letting a nonsense `age` feature reach the model,
        # where it would silently distort the prediction instead of
        # loudly failing.
        today = date.today()
        if v >= today:
            raise ValueError("dob must be in the past.")
        age_years = (today - v).days / 365.25
        if age_years < 13 or age_years > 120:
            raise ValueError(f"dob implies an implausible age ({age_years:.1f} years).")
        return v

    @field_validator("trans_time")
    @classmethod
    def trans_time_must_be_reasonable(cls, v: datetime) -> datetime:
        # Normalize to UTC so downstream arithmetic (age-at-transaction-time,
        # hour/day-of-week extraction) is never silently offset by an
        # unstated timezone assumption.
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        # Allow a small forward skew for clock drift between the merchant's
        # server and ours; anything beyond that is more likely a bad
        # integration (e.g. sending a client-side clock, or a bug that
        # sends a far-future placeholder date) than a real transaction.
        if (v - now).total_seconds() > 300:
            raise ValueError("trans_time is too far in the future to be a real transaction.")
        return v

    @model_validator(mode="after")
    def dob_must_precede_transaction(self) -> "TransactionInput":
        # A cardholder cannot transact before they were born. This is a
        # cross-field check, so it has to live in a model_validator rather
        # than a single-field field_validator.
        if self.dob >= self.trans_time.date():
            raise ValueError("dob must be before trans_time.")
        return self


class PredictionOutput(BaseModel):
    """Response returned by POST /v1/predict-fraud."""

    is_fraud: Optional[bool] = Field(
        None, description="Model verdict at the cost-optimal binary threshold. "
                           "None if scoring failed and a fallback decision was used instead.",
    )
    fraud_probability: Optional[float] = Field(
        None, ge=0, le=1,
        description="Model's predicted probability of fraud. None if scoring failed.",
    )
    action_taken: str = Field(
        ..., description="One of 'BLOCK', 'REVIEW', 'APPROVE' — the operational decision.",
    )
    shap_explanation: dict[str, float] = Field(
        default_factory=dict,
        description="Top 3 contributing features and their SHAP contribution values for "
                    "this specific transaction. Empty if scoring failed.",
    )
    threshold_used: Optional[float] = Field(
        None, description="The decision threshold applied to fraud_probability. "
                           "None if a fallback decision was used instead of the model.",
    )
    fallback_triggered: bool = Field(
        False, description="True if this response came from the rules-based fallback "
                            "path rather than the ML model (e.g. due to malformed input "
                            "or an internal scoring error).",
    )
    reason: Optional[str] = Field(
        None, description="Human-readable explanation, populated mainly on the fallback path.",
    )