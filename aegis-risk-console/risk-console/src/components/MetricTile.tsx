import { type ReactNode } from "react";

import { cn } from "@/lib/utils";

export type MetricTone = "neutral" | "approve" | "review" | "block";

interface MetricTileProps {
  label: string;
  /** Pre-null-checked value. Pass null when the backend didn't supply one. */
  value: number | null;
  /** How to render a non-null value. Percent multiplies by 100. */
  format?: "percent" | "decimal";
  precision?: number;
  caption?: string;
  tone?: MetricTone;
  icon?: ReactNode;
}

const toneText: Record<MetricTone, string> = {
  neutral: "text-ink",
  approve: "text-signal-approve",
  review: "text-signal-review",
  block: "text-signal-block",
};

function formatValue(value: number | null, format: "percent" | "decimal", precision: number) {
  if (value === null || Number.isNaN(value)) return "—";
  if (format === "percent") return `${(value * 100).toFixed(precision)}%`;
  return value.toFixed(precision);
}

export function MetricTile({
  label,
  value,
  format = "decimal",
  precision = 2,
  caption,
  tone = "neutral",
  icon,
}: MetricTileProps) {
  const display = formatValue(value, format, precision);
  const isMissing = value === null || Number.isNaN(value);

  return (
    <div className="rounded-lg border border-hairline bg-surface-raised px-4 py-3.5">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
          {label}
        </span>
        {icon ? <span className="text-ink-faint">{icon}</span> : null}
      </div>
      <div
        className={cn(
          "mt-1.5 font-mono text-3xl font-medium leading-none tabular-nums",
          isMissing ? "text-ink-faint" : toneText[tone],
        )}
      >
        {display}
      </div>
      {caption ? (
        <p className="mt-1.5 text-[0.72rem] text-ink-faint">{caption}</p>
      ) : null}
    </div>
  );
}
