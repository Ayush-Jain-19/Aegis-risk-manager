import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipContentProps } from "recharts";

const MAX_FEATURES = 8;
const FRAUD_COLOR = "#DD6660";
const LEGIT_COLOR = "#4FAE8A";

interface ShapChartProps {
  shapExplanation: Record<string, number>;
}

interface ShapRow {
  feature: string;
  value: number;
}

function formatFeatureName(key: string): string {
  return key.replace(/[_-]+/g, " ").trim().toUpperCase();
}

function formatSignedValue(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "\u2212" : "";
  return `${sign}${Math.abs(value).toFixed(3)}`;
}

function ShapTooltip({ active, payload }: TooltipContentProps) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0]?.payload as ShapRow | undefined;
  if (!row) return null;

  const pushesToward = row.value >= 0 ? "FRAUD" : "LEGITIMATE";
  const color = row.value >= 0 ? FRAUD_COLOR : LEGIT_COLOR;

  return (
    <div className="rounded-md border border-hairline-strong bg-surface-raised px-3 py-2 text-xs shadow-panel">
      <p className="font-mono uppercase tracking-wide text-ink-muted">{row.feature}</p>
      <p className="mt-1 font-mono text-sm tabular-nums" style={{ color }}>
        {formatSignedValue(row.value)}
      </p>
      <p className="mt-0.5 text-[0.68rem] text-ink-faint">Pushes toward {pushesToward}</p>
    </div>
  );
}

export function ShapChart({ shapExplanation }: ShapChartProps) {
  const rows = useMemo<ShapRow[]>(() => {
    const entries = Object.entries(shapExplanation ?? {})
      .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
      .map(([feature, value]) => ({ feature: formatFeatureName(feature), value }));

    entries.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return entries.slice(0, MAX_FEATURES);
  }, [shapExplanation]);

  if (rows.length === 0) {
    return (
      <div className="flex h-40 flex-col items-center justify-center gap-1 rounded-md border border-dashed border-hairline text-center">
        <p className="text-sm text-ink-muted">No feature attributions were returned.</p>
        <p className="text-xs text-ink-faint">The model may not have produced a SHAP explanation for this transaction.</p>
      </div>
    );
  }

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.value)), 0.001);
  const domainPad = maxAbs * 1.35;
  const chartHeight = Math.max(rows.length * 42, 160);

  return (
    <div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 4, right: 46, bottom: 4, left: 4 }}
          barCategoryGap={14}
        >
          <XAxis type="number" domain={[-domainPad, domainPad]} hide />
          <YAxis
            type="category"
            dataKey="feature"
            width={132}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#8D93A3", fontFamily: "IBM Plex Mono, monospace", fontSize: 11 }}
          />
          <ReferenceLine x={0} stroke="#383F4D" strokeWidth={1} />
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.03)" }} content={ShapTooltip} />
          <Bar
            dataKey="value"
            radius={3}
            background={{ fill: "rgba(255,255,255,0.02)", radius: 3 }}
            isAnimationActive
            animationDuration={650}
            animationEasing="ease-out"
          >
            {rows.map((row) => (
              <Cell key={row.feature} fill={row.value >= 0 ? FRAUD_COLOR : LEGIT_COLOR} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(value: unknown) => formatSignedValue(value as number)}
              style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 11, fill: "#8D93A3" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-3 flex items-center justify-center gap-6 border-t border-hairline pt-3">
        <span className="flex items-center gap-1.5 font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
          <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: LEGIT_COLOR }} />
          Toward legitimate
        </span>
        <span className="flex items-center gap-1.5 font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
          <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: FRAUD_COLOR }} />
          Toward fraud
        </span>
      </div>
    </div>
  );
}
