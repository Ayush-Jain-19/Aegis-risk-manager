import { useState } from "react";
import { motion } from "framer-motion";
import { Gauge, Percent, RadioTower, WifiOff } from "lucide-react";

import { AppHeader } from "@/components/AppHeader";
import { DecisionBanner } from "@/components/DecisionBanner";
import { MetricTile, type MetricTone } from "@/components/MetricTile";
import { ShapChart } from "@/components/ShapChart";
import { TransactionForm } from "@/components/TransactionForm";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { predictFraud } from "@/lib/api";
import type { RiskDecision, TransactionPayload } from "@/lib/types";

const ACTION_TONE: Record<RiskDecision["actionTaken"], MetricTone> = {
  APPROVE: "approve",
  REVIEW: "review",
  BLOCK: "block",
};

function EmptyState() {
  return (
    <Card className="flex min-h-[420px] flex-col items-center justify-center gap-3 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-full border border-hairline-strong bg-surface-raised text-wire">
        <RadioTower className="h-5 w-5" strokeWidth={1.75} />
      </span>
      <div>
        <p className="font-display text-lg text-ink">Awaiting a transaction</p>
        <p className="mx-auto mt-1 max-w-sm text-sm text-ink-faint">
          Submit the form to score a transaction. The decision, probability, and feature
          attributions will render here.
        </p>
      </div>
    </Card>
  );
}

function LoadingState() {
  return (
    <Card className="min-h-[420px] p-6">
      <div className="animate-pulse space-y-5">
        <div className="h-20 rounded-lg bg-surface-raised" />
        <div className="grid grid-cols-2 gap-4">
          <div className="h-20 rounded-lg bg-surface-raised" />
          <div className="h-20 rounded-lg bg-surface-raised" />
        </div>
        <div className="h-56 rounded-lg bg-surface-raised" />
      </div>
    </Card>
  );
}

function ConnectionErrorState({ message }: { message: string }) {
  return (
    <Card className="border-signal-block-border bg-signal-block-bg">
      <CardContent className="flex items-start gap-3 pt-5">
        <WifiOff className="mt-0.5 h-5 w-5 shrink-0 text-signal-block" strokeWidth={1.75} />
        <div>
          <p className="font-display text-base font-medium text-signal-block">
            Could not score the transaction
          </p>
          <p className="mt-1 text-sm text-ink-muted">{message}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function RiskConsole() {
  const [decision, setDecision] = useState<RiskDecision | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(payload: TransactionPayload) {
    setIsSubmitting(true);
    setConnectionError(null);
    const outcome = await predictFraud(payload);
    setIsSubmitting(false);

    if (outcome.ok) {
      setDecision(outcome.decision);
    } else {
      setConnectionError(outcome.message);
    }
  }

  return (
    <div className="min-h-screen bg-canvas">
      <AppHeader />

      <main className="container max-w-7xl py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <TransactionForm onSubmit={handleSubmit} isSubmitting={isSubmitting} />
          </div>

          <div className="space-y-5 lg:col-span-7">
            {isSubmitting && <LoadingState />}
            {!isSubmitting && connectionError && <ConnectionErrorState message={connectionError} />}
            {!isSubmitting && !connectionError && !decision && <EmptyState />}

            {!isSubmitting && !connectionError && decision && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25 }}
                className="space-y-5"
              >
                <DecisionBanner decision={decision} />

                {!decision.fallbackTriggered && (
                  <>
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                      <MetricTile
                        label="Fraud probability"
                        value={decision.fraudProbability}
                        format="percent"
                        precision={2}
                        tone={ACTION_TONE[decision.actionTaken]}
                        icon={<Percent className="h-3.5 w-3.5" />}
                        caption="Model-estimated likelihood this transaction is fraudulent."
                      />
                      <MetricTile
                        label="Decision threshold"
                        value={decision.thresholdUsed}
                        format="decimal"
                        precision={3}
                        tone="neutral"
                        icon={<Gauge className="h-3.5 w-3.5" />}
                        caption="Probability cutoff configured for automatic action."
                      />
                    </div>

                    <Card>
                      <CardHeader>
                        <CardTitle>Feature attribution</CardTitle>
                        <p className="text-xs text-ink-faint">
                          SHAP values ranked by influence — how each feature pushed this score.
                        </p>
                      </CardHeader>
                      <CardContent>
                        <ShapChart shapExplanation={decision.shapExplanation} />
                      </CardContent>
                    </Card>
                  </>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
