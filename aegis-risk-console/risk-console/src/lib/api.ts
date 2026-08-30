import { parseApiResponse, type ApiOutcome, type TransactionPayload } from "@/lib/types";

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

const PREDICT_ENDPOINT = `${API_BASE_URL}/v1/predict-fraud`;

/**
 * Submits a transaction for scoring. This never throws and never rejects —
 * every failure mode (network down, non-JSON body, unexpected shape, 4xx/5xx)
 * resolves to an ApiOutcome so the console always has something safe to
 * render.
 */
export async function predictFraud(payload: TransactionPayload): Promise<ApiOutcome> {
  let response: Response;
  try {
    response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    return {
      ok: false,
      message:
        "Could not reach the risk engine. Confirm the API is running at " +
        API_BASE_URL +
        ".",
    };
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON body (proxy error page, empty response, etc). Fall through
    // with body === null; parseApiResponse handles this as "unparseable".
  }

  // Both the 200 success schema and the 422/500 fallback schema are valid,
  // meaningful responses — parseApiResponse figures out which one it is.
  // A body that matches neither still degrades into a safe REVIEW decision.
  const decision = parseApiResponse(body);
  return { ok: true, decision };
}

export type ConnectivityStatus = "checking" | "online" | "offline";

/**
 * Best-effort reachability probe for the header's system-status pill. Treats
 * any response — including a 404 — as "the server answered", since the goal
 * is distinguishing "API is up" from "network/host unreachable", not
 * validating a specific health route the contract doesn't define.
 */
export async function checkApiConnectivity(signal?: AbortSignal): Promise<ConnectivityStatus> {
  try {
    await fetch(API_BASE_URL, { method: "GET", signal });
    return "online";
  } catch {
    return "offline";
  }
}
