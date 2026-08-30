import { z } from "zod";

/**
 * ---------------------------------------------------------------------------
 * Outbound: transaction submission payload
 * ---------------------------------------------------------------------------
 */
export interface TransactionPayload {
  amount: number;
  merchant: string;
  category: string;
  merchant_lat: number;
  merchant_long: number;
  customer_lat: number;
  customer_long: number;
  dob: string; // ISO date, e.g. "1990-05-14"
  trans_time: string; // ISO datetime, e.g. "2026-08-29T12:00:00"
}

export const transactionPayloadSchema = z.object({
  amount: z.number().positive("must be greater than 0"),
  merchant: z.string().min(1, "required"),
  category: z.string().min(1, "required"),
  merchant_lat: z.number().min(-90).max(90),
  merchant_long: z.number().min(-180).max(180),
  customer_lat: z.number().min(-90).max(90),
  customer_long: z.number().min(-180).max(180),
  dob: z.string().min(1, "required"),
  trans_time: z.string().min(1, "required"),
});

/**
 * Parses developer-supplied raw JSON into a TransactionPayload. Returns a
 * human-readable error string (never a stack trace) on failure so the raw
 * JSON editor can surface it inline.
 */
export function parseTransactionPayload(
  raw: string,
): { payload: TransactionPayload | null; error: string | null } {
  if (!raw.trim()) {
    return { payload: null, error: "Enter a JSON transaction object." };
  }

  let candidate: unknown;
  try {
    candidate = JSON.parse(raw);
  } catch {
    return { payload: null, error: "Invalid JSON — check for a trailing comma or unquoted key." };
  }

  const result = transactionPayloadSchema.safeParse(candidate);
  if (!result.success) {
    const [firstIssue] = result.error.issues;
    const path = firstIssue?.path.join(".") || "payload";
    return { payload: null, error: `${path}: ${firstIssue?.message ?? "invalid value"}` };
  }

  return { payload: result.data, error: null };
}

export const ACTIONS = ["APPROVE", "REVIEW", "BLOCK"] as const;
export type ActionTaken = (typeof ACTIONS)[number];

function isActionTaken(value: unknown): value is ActionTaken {
  return typeof value === "string" && (ACTIONS as readonly string[]).includes(value);
}

/**
 * ---------------------------------------------------------------------------
 * Inbound: the two shapes the API can return.
 * Every field is treated as untrusted — the backend is production-grade, but
 * the console must never trust a wire payload blindly, model drift, a proxy
 * timeout, or a partial JSON body should degrade gracefully, not crash.
 * ---------------------------------------------------------------------------
 */
const shapExplanationSchema = z.record(z.string(), z.number()).default({});

const successResponseSchema = z.object({
  is_fraud: z.boolean().nullable().default(null),
  fraud_probability: z.number().nullable().default(null),
  action_taken: z.string().default("REVIEW"),
  shap_explanation: shapExplanationSchema,
  threshold_used: z.number().nullable().default(null),
  fallback_triggered: z.boolean().default(false),
  reason: z.string().nullable().default(null),
});

const validationErrorSchema = z.object({
  field: z.string().default("unknown"),
  message: z.string().default("No further detail provided."),
  type: z.string().default("error"),
});

const fallbackResponseSchema = z.object({
  status: z.string().default("error"),
  validation_errors: z.array(validationErrorSchema).default([]),
  decision: z.object({
    is_fraud: z.boolean().nullable().default(null),
    fraud_probability: z.number().nullable().default(null),
    action_taken: z.string().default("REVIEW"),
    shap_explanation: shapExplanationSchema,
    threshold_used: z.number().nullable().default(null),
    fallback_triggered: z.boolean().default(true),
    reason: z.string().nullable().default(null),
  }),
});

/**
 * ---------------------------------------------------------------------------
 * Normalized shape — every component downstream renders from this, and only
 * this. It carries no knowledge of which wire shape it came from.
 * ---------------------------------------------------------------------------
 */
export interface ValidationIssue {
  field: string;
  message: string;
  type: string;
}

export interface RiskDecision {
  /** Where this decision came from — informs the UI's framing, not its logic. */
  origin: "model" | "fallback";
  actionTaken: ActionTaken;
  isFraud: boolean | null;
  fraudProbability: number | null;
  thresholdUsed: number | null;
  shapExplanation: Record<string, number>;
  fallbackTriggered: boolean;
  reason: string | null;
  validationErrors: ValidationIssue[];
}

function toRiskDecision(
  origin: "model" | "fallback",
  data: {
    is_fraud: boolean | null;
    fraud_probability: number | null;
    action_taken: string;
    shap_explanation: Record<string, number>;
    threshold_used: number | null;
    fallback_triggered: boolean;
    reason: string | null;
  },
  validationErrors: ValidationIssue[] = [],
): RiskDecision {
  return {
    origin,
    actionTaken: isActionTaken(data.action_taken) ? data.action_taken : "REVIEW",
    isFraud: data.is_fraud,
    fraudProbability: data.fraud_probability,
    thresholdUsed: data.threshold_used,
    shapExplanation: data.shap_explanation ?? {},
    fallbackTriggered: data.fallback_triggered,
    reason: data.reason,
    validationErrors,
  };
}

export type ApiOutcome =
  | { ok: true; decision: RiskDecision }
  | { ok: false; message: string };

/**
 * Parses an arbitrary, untrusted JSON body from the /v1/predict-fraud
 * endpoint into a RiskDecision. Falls through three tiers:
 *   1. Success schema (200 OK, model scored the transaction)
 *   2. Fallback schema (422 / 500, rules engine took over)
 *   3. An explicit "unparseable" decision that routes to REVIEW rather than
 *      ever letting a malformed body reach the render tree.
 */
export function parseApiResponse(raw: unknown): RiskDecision {
  const success = successResponseSchema.safeParse(raw);
  if (success.success) {
    return toRiskDecision("model", success.data);
  }

  const fallback = fallbackResponseSchema.safeParse(raw);
  if (fallback.success) {
    return toRiskDecision(
      "fallback",
      fallback.data.decision,
      fallback.data.validation_errors,
    );
  }

  // Neither shape matched. Rather than crash, route to manual review with a
  // conservative, honest message — this is the same posture the backend
  // itself takes when it can't be sure.
  return toRiskDecision("fallback", {
    is_fraud: null,
    fraud_probability: null,
    action_taken: "REVIEW",
    shap_explanation: {},
    threshold_used: null,
    fallback_triggered: true,
    reason:
      "The API response did not match an expected shape. Routed to manual review as a precaution.",
  });
}
