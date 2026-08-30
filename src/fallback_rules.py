"""
src/fallback_rules.py

A true high-availability fallback: applied when the ML pipeline itself
fails (missing model file, corrupted artifact, out-of-memory, prediction
timeout, an unexpected SHAP error — anything), for a request that already
passed input validation and therefore contains clean, trustworthy
transaction data. This is deliberately distinct from the fallback in
exception_handlers.py:

  - exception_handlers.py's `dummy_fallback_decision` fires when the
    REQUEST ITSELF is malformed (failed Pydantic validation) — we don't
    trust the data enough to reason about it at all, so it always returns
    "REVIEW" with no attempt at business logic.
  - THIS module fires when the request was fine but the ML SYSTEM broke —
    we have good data and no working model, so we apply real (if simple)
    risk logic instead of defaulting to the same answer every time.

WHY THIS FILE HAS ALMOST NO DEPENDENCIES (BY DESIGN, NOT OVERSIGHT)
-----------------------------------------------------------------------
This module intentionally does NOT import joblib, yaml, pandas, the
project's config loader, or anything from api/main.py's ModelArtifacts.
The entire point of a fallback is to keep working when something else
breaks — if apply_fallback_rules() depended on config.yaml parsing
correctly, or on a shared utility module that also happens to be involved
in loading the model, a failure with a broad enough blast radius (a bad
deploy, a corrupted config file, a disk issue) could take down the
fallback at the same moment it takes down the primary system, which is
exactly the single point of failure a fallback exists to eliminate. The
thresholds below are therefore hardcoded module-level constants, not
config-driven — a deliberate trade-off of flexibility for independence.

WHY THE FUNCTION CATCHES ITS OWN EXCEPTIONS
------------------------------------------------
A fallback that can itself throw is not a fallback — it's just another
thing that can go wrong at the worst possible moment (while the primary
system is already down). Every piece of data extraction here is
defensive, and the function's outer layer catches anything unexpected
regardless, always returning a well-formed response.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Hardcoded on purpose — see module docstring. Tune these by editing this
# file directly (and redeploying), not via a config file this module
# should never depend on.
HIGH_RISK_HOUR_START = 0     # 12 AM
HIGH_RISK_HOUR_END = 5       # 5 AM (exclusive) — window is [0, 5)
HIGH_RISK_AMOUNT_THRESHOLD = 1000.0
CATASTROPHIC_AMOUNT_CEILING = 5000.0


def apply_fallback_rules(transaction_data: dict[str, Any]) -> dict[str, Any]:
    """
    Applies simple, defensible risk rules to a transaction when the ML
    pipeline is unavailable, and returns a dict matching the shape of
    `PredictionOutput` (src/api/schemas.py) so the caller can construct
    that response directly.

    Rules, in order (first match wins):
      1. amount > $5,000 (CATASTROPHIC_AMOUNT_CEILING) -> BLOCK, regardless
         of time. Rationale: during a genuine ML outage, letting an
         unscored, very large transaction through unconditionally is a
         worse business outcome than a false decline a legitimate
         high-spending customer can resolve with a phone call.
      2. amount > $1,000 (HIGH_RISK_AMOUNT_THRESHOLD) AND the transaction
         hour falls in [12 AM, 5 AM) -> REVIEW. This is the exact
         "high amount + late night" heuristic classically associated with
         elevated fraud risk, used here as a coarse stand-in for what the
         model would normally assess.
      3. Otherwise -> APPROVE. The conservative BLOCK/REVIEW tiers above
         already capture the patterns most worth being cautious about;
         defaulting everything else to APPROVE keeps checkout friction low
         during a degraded-mode window rather than blocking the vast
         majority of ordinary transactions.

    `transaction_data` is expected to look like a validated TransactionInput
    dumped to a dict (i.e. has "amount" and "trans_time" keys), but every
    field access below is defensive — this function must never raise,
    even given a malformed or incomplete dict, since a failure here would
    leave the request with no answer at all.
    """
    try:
        amount = _safe_float(transaction_data.get("amount"))
        hour = _extract_hour(transaction_data.get("trans_time"))
        action, reason = _decide_action(amount, hour)
    except Exception as exc:  # noqa: BLE001 — intentionally blanket; see module docstring
        # This branch should be unreachable given the defensive helpers
        # below, but a fallback mechanism earns its "high availability"
        # label by assuming it will be reached anyway. If it somehow is,
        # log it as critical (this represents a bug in the LAST line of
        # defense) and still return a safe, well-formed response rather
        # than letting the exception propagate.
        logger.critical(
            "apply_fallback_rules() itself raised an unexpected error (%s) — "
            "this should never happen. Defaulting to the safest possible "
            "response (REVIEW).",
            exc, exc_info=True,
        )
        action = "REVIEW"
        reason = "The fallback rules engine encountered an internal error; defaulted to REVIEW as the safest possible outcome."

    logger.warning(
        "Rules-based fallback triggered (ML pipeline unavailable): action=%s, reason=%s",
        action, reason,
    )

    return {
        "is_fraud": None,
        "fraud_probability": None,
        "action_taken": action,
        "shap_explanation": {},
        "threshold_used": None,
        "fallback_triggered": True,
        "reason": reason,
    }


def _decide_action(amount: float, hour: int | None) -> tuple[str, str]:
    if amount > CATASTROPHIC_AMOUNT_CEILING:
        return "BLOCK", (
            f"ML model unavailable; fallback rule triggered: amount (${amount:,.2f}) "
            f"exceeds the catastrophic-loss ceiling (${CATASTROPHIC_AMOUNT_CEILING:,.2f}) "
            f"— blocked outright as a conservative default."
        )

    is_high_risk_hour = hour is not None and HIGH_RISK_HOUR_START <= hour < HIGH_RISK_HOUR_END
    if amount > HIGH_RISK_AMOUNT_THRESHOLD and is_high_risk_hour:
        return "REVIEW", (
            f"ML model unavailable; fallback rule triggered: amount (${amount:,.2f}) "
            f"exceeds ${HIGH_RISK_AMOUNT_THRESHOLD:,.2f} during a high-risk overnight "
            f"window (hour={hour}) — routed to manual review."
        )

    return "APPROVE", (
        "ML model unavailable; transaction did not match any high-risk pattern under "
        "the rules-based fallback policy — approved to minimize checkout friction "
        "during degraded-mode operation."
    )


def _safe_float(value: Any) -> float:
    """Never raises: coerces to float, falling back to 0.0 for anything
    that can't be interpreted as a number (None, a bad string, etc.).
    Treating an unparseable amount as 0 is deliberately conservative in
    the SAFE direction here — it can only route toward APPROVE via the
    amount-based rules, never toward incorrectly blocking a transaction
    whose amount we simply failed to parse; the catastrophic-ceiling and
    high-risk-hour rules above are the ones actually carrying the
    protective weight in this fallback."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        logger.warning("Fallback could not parse transaction amount (%r); treating as 0.0.", value)
        return 0.0


def _extract_hour(trans_time: Any) -> int | None:
    """Best-effort extraction of the hour-of-day. Returns None (rather
    than raising) for anything that isn't a datetime or a parseable
    ISO-8601 string — in that case, `_decide_action` simply can't apply
    the time-based rule and falls through to the amount-only rules."""
    if trans_time is None:
        return None
    if isinstance(trans_time, datetime):
        return trans_time.hour
    if isinstance(trans_time, str):
        try:
            return datetime.fromisoformat(trans_time.replace("Z", "+00:00")).hour
        except ValueError:
            logger.warning("Fallback could not parse trans_time (%r) to extract hour.", trans_time)
            return None
    return None