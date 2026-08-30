"""
src/api/exception_handlers.py

WHY THIS FILE EXISTS
---------------------
"Graceful failure" for a fraud API means two things at once, and both are
implemented here:
  1. The merchant integration calling us gets a clear, structured error —
     not a raw 500 stack trace — telling them EXACTLY which field failed
     and why, so they can fix their integration.
  2. The transaction itself still gets SOME decision. A merchant checkout
     flow generally can't just hang waiting for us to fix a bug — if our
     validation or scoring pipeline fails, we still owe them an action.
     That's the "dummy fallback rules-based decision" requirement: when we
     can't confidently score a transaction, we don't guess "APPROVE" (that
     would defeat the entire purpose of having a fraud layer) — we default
     to "REVIEW", handing the ambiguous case to a human rather than
     silently taking on risk or silently blocking a legitimate customer.

Both handlers here return the SAME response shape (see PredictionOutput in
schemas.py) that a successful prediction would — just with
`fallback_triggered=True`, `is_fraud=None`, and `fraud_probability=None` —
so a merchant's integration only needs to parse one response shape,
whether the request succeeded or failed.
"""

from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def dummy_fallback_decision(reason: str) -> dict:
    """
    Rules-based fallback used whenever the ML pipeline cannot be trusted
    to produce a scored decision (malformed input, missing critical
    feature, or an unexpected scoring error).

    WHY "REVIEW" AND NOT "BLOCK" OR "APPROVE":
      - APPROVE would silently let a potentially fraudulent transaction
        through with zero scrutiny — exactly the failure mode this whole
        system exists to prevent. Never the right default on a failure path.
      - BLOCK would auto-decline every transaction that merely tripped a
        validation bug (e.g. a merchant integration sending dob as
        "01/15/1990" instead of ISO-8601) — that's real revenue and real
        customer friction lost to a software bug, not to actual fraud risk.
      - REVIEW routes the ambiguous case to a human analyst, which is the
        conservative, defensible middle ground: no unscored risk is taken
        on, and no legitimate customer is unilaterally declined by a
        pipeline that couldn't even validate their request.

    This is intentionally a simple, fixed-rule function ("dummy" per the
    requirement) rather than a second model — a fallback path that itself
    depends on the failing pipeline would defeat the purpose of a fallback.
    A fuller rules engine (velocity checks, merchant risk lists, amount
    ceilings) belongs in its own src/fallback_rules.py module; this stub
    keeps the exception-handling path self-contained and dependency-free.
    """
    return {
        "is_fraud": None,
        "fraud_probability": None,
        "action_taken": "REVIEW",
        "shap_explanation": {},
        "threshold_used": None,
        "fallback_triggered": True,
        "reason": reason,
    }


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handles Pydantic/FastAPI request validation failures — e.g. a missing
    field, a negative `amount`, an unparsable `trans_time`. Returns HTTP
    422 (not 400 or 500) since the request was syntactically valid JSON
    but semantically invalid per our schema — the standard HTTP status for
    exactly this case.

    The response body deliberately echoes back precisely which field(s)
    failed and why (`exc.errors()`), because "graceful failure" for an API
    consumer means they can programmatically distinguish "you sent me a
    negative amount" from "you forgot the dob field" without guessing from
    a generic message.
    """
    error_details = [
        {
            # exc.errors()[i]["loc"] is a tuple like ("body", "amount") —
            # we drop the leading "body" element since it's an internal
            # FastAPI implementation detail the merchant's integration
            # team doesn't need to know about.
            "field": ".".join(str(loc) for loc in err["loc"] if loc != "body"),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]

    logger.warning(
        "Validation failed for request to %s: %s",
        request.url.path, error_details,
    )

    fallback = dummy_fallback_decision(
        reason="Input validation failed — see 'validation_errors' for details. "
               "Routed to manual review since the transaction could not be safely scored.",
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "validation_error",
            "validation_errors": error_details,
            "decision": fallback,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches anything NOT already handled — a missing critical feature that
    slipped past schema validation, a model/encoder loading issue, an
    unexpected error inside feature engineering or SHAP. This is the
    outermost safety net: it guarantees the API process itself never
    crashes and never returns a bare stack trace to a merchant integration,
    no matter what goes wrong inside the scoring pipeline.

    WHY 500, not 422 here: unlike validation errors, this branch means the
    REQUEST was well-formed — the failure is on our side (a bug, an
    unavailable dependency, an unhandled edge case), so the correct status
    code is 500, while the body still carries the same safe fallback
    decision so the caller isn't left without an actionable response.
    """
    logger.error(
        "Unhandled exception while processing request to %s: %s",
        request.url.path, str(exc), exc_info=True,
    )

    fallback = dummy_fallback_decision(
        reason="An internal error occurred while scoring this transaction. "
               "Routed to manual review as a precaution.",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "internal_error",
            "decision": fallback,
        },
    )


def register_exception_handlers(app) -> None:
    """Wires both handlers into the FastAPI app. Called once from main.py
    at app-creation time, kept in one place so it's obvious at a glance
    that these two failure modes are covered — no exception handler
    registration should be scattered elsewhere in the codebase."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("Registered custom validation and generic exception handlers.")