# Aegis — Risk Console

A production-grade frontend for an AI fraud-detection API: a FastAPI backend
scores a transaction and this console renders the decision, the probability,
and a SHAP-based explanation of *why*.

Built with React + Vite + strict TypeScript, Tailwind, hand-built shadcn/ui
(Radix) primitives, Recharts, and Framer Motion — no Streamlit, no plain HTML.

## Design system

An editorial, dark, fintech-terminal aesthetic — rich charcoal/slate surfaces
(never flat black), a restrained steel-blue accent, and functional signal
colors (muted emerald / amber / brick-red) that carry the APPROVE / REVIEW /
BLOCK verdicts. Type pairs a **Fraunces** display serif (headlines, brand
mark) against **Inter** for UI copy and **IBM Plex Mono** for every number —
amounts, coordinates, timestamps, and SHAP values all render in tabular
figures, like a ledger.

The SHAP chart is the signature element: a diverging "risk ledger" — feature
names on the left, a zero-line splitting the plot, bars extending toward
FRAUD (red, right) or LEGITIMATE (green, left), values printed at each tip.

## Getting started

```bash
npm install
npm run dev
```

The app expects the FastAPI backend at `http://127.0.0.1:8000` by default.
To point at a different host, copy `.env.example` to `.env` and set
`VITE_API_BASE_URL`.

```bash
npm run build     # type-check (tsc -b) + production build to dist/
npm run preview   # serve the production build locally
npm run lint       # oxlint
```

## Project structure

```
src/
  lib/
    types.ts        Zod schemas for both API response shapes + the
                     normalized RiskDecision every component renders from
    api.ts           predictFraud() — never throws, always resolves
    utils.ts         cn() className helper
  components/
    ui/              Hand-built shadcn-style primitives (button, card,
                     input, select, switch, tabs, ...)
    AppHeader.tsx    Brand mark + live API connectivity pill
    TransactionForm.tsx   Guided multi-column form, with a dev-mode
                          toggle to RawJsonEditor.tsx
    RawJsonEditor.tsx     Raw JSON payload editor with live validation
    DecisionBanner.tsx    APPROVE / REVIEW / BLOCK verdict banner —
                          surfaces the fallback reason prominently when
                          the rules engine takes over
    MetricTile.tsx        Tabular-numeral metric cards
    ShapChart.tsx         The diverging SHAP "risk ledger" chart
  pages/
    RiskConsole.tsx  Composes everything; owns request/loading/error state
```

## How the API contract is handled defensively

Every field in both the 200 success response and the 422/500 fallback
response is treated as untrusted. `lib/types.ts` runs both shapes through
Zod schemas with defaults for every field, and normalizes whichever one
matches into a single `RiskDecision`. A response that matches *neither*
shape (a network hiccup, a malformed proxy body, an unannounced backend
change) never crashes the UI — it degrades into an explicit REVIEW decision
with an honest explanation instead.

When `fallback_triggered` is `true`, the console hides the fraud-probability
metric and the SHAP chart entirely (the model didn't run, so there's nothing
real to show) and renders `decision.reason` prominently inside the banner so
it's clear the high-availability rules engine — not the model — made the
call.
