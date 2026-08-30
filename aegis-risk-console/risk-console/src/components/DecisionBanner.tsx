import { motion } from "framer-motion";
import { AlertTriangle, ShieldAlert, ShieldCheck, ShieldX } from "lucide-react";

import type { ActionTaken, RiskDecision } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DecisionBannerProps {
  decision: RiskDecision;
}

interface ActionCopy {
  headline: string;
  description: string;
  icon: typeof ShieldCheck;
  containerClass: string;
  iconClass: string;
  headlineClass: string;
}

const ACTION_COPY: Record<ActionTaken, ActionCopy> = {
  APPROVE: {
    headline: "Transaction approved",
    description: "Scored below the decision threshold. No manual action required.",
    icon: ShieldCheck,
    containerClass: "border-signal-approve-border bg-signal-approve-bg",
    iconClass: "text-signal-approve",
    headlineClass: "text-signal-approve",
  },
  REVIEW: {
    headline: "Flagged for review",
    description: "Risk score falls in the review band. Route to a manual analyst.",
    icon: ShieldAlert,
    containerClass: "border-signal-review-border bg-signal-review-bg",
    iconClass: "text-signal-review",
    headlineClass: "text-signal-review",
  },
  BLOCK: {
    headline: "Transaction blocked",
    description: "Risk score exceeded the block threshold. Transaction stopped automatically.",
    icon: ShieldX,
    containerClass: "border-signal-block-border bg-signal-block-bg",
    iconClass: "text-signal-block",
    headlineClass: "text-signal-block",
  },
};

export function DecisionBanner({ decision }: DecisionBannerProps) {
  const copy = ACTION_COPY[decision.actionTaken];
  const Icon = copy.icon;

  return (
    <motion.div
      key={`${decision.actionTaken}-${decision.origin}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={cn("rounded-xl border p-5 sm:p-6", copy.containerClass)}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-4">
          <span
            className={cn(
              "mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-canvas/40",
              copy.iconClass,
            )}
          >
            <Icon className="h-6 w-6" strokeWidth={1.75} />
          </span>
          <div>
            <h2
              className={cn(
                "font-display text-2xl font-medium tracking-tight sm:text-[1.75rem]",
                copy.headlineClass,
              )}
            >
              {copy.headline}
            </h2>
            <p className="mt-1 max-w-xl text-sm text-ink-muted">{copy.description}</p>
          </div>
        </div>

        <span className="self-start whitespace-nowrap rounded-sm border border-hairline-strong bg-canvas/40 px-2.5 py-1 font-mono text-[0.7rem] uppercase tracking-wide text-ink-muted sm:self-center">
          {decision.origin === "model" ? "Model decision" : "Rules engine"}
        </span>
      </div>

      {decision.fallbackTriggered ? (
        <div className="mt-4 flex items-start gap-3 rounded-md border border-hairline-strong bg-canvas/50 p-3.5">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-signal-review" strokeWidth={1.75} />
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-signal-review">
              High-availability rules engine engaged
            </p>
            <p className="mt-1 text-sm text-ink-muted">
              {decision.reason ?? "The model was unavailable; a conservative fallback rule set made this decision."}
            </p>
            {decision.validationErrors.length > 0 ? (
              <ul className="mt-2 space-y-1 border-t border-hairline pt-2">
                {decision.validationErrors.map((issue, i) => (
                  <li key={`${issue.field}-${i}`} className="font-mono text-xs text-ink-faint">
                    <span className="text-ink-muted">{issue.field}</span> — {issue.message}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </div>
      ) : null}
    </motion.div>
  );
}
