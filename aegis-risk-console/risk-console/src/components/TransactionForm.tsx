import { useState, type FormEvent, type ReactNode } from "react";
import { Braces, ListTree, Loader2, SlidersHorizontal } from "lucide-react";

import { RawJsonEditor } from "@/components/RawJsonEditor";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import type { TransactionPayload } from "@/lib/types";

const CATEGORIES = [
  "shopping_net",
  "shopping_pos",
  "grocery_net",
  "grocery_pos",
  "gas_transport",
  "misc_net",
  "misc_pos",
  "entertainment",
  "food_dining",
  "health_fitness",
  "home",
  "kids_pets",
  "personal_care",
  "travel",
] as const;

const SAMPLE_PAYLOAD: TransactionPayload = {
  amount: 249.99,
  merchant: "fraud_Kutch_and_Sons",
  category: "shopping_net",
  merchant_lat: 40.7128,
  merchant_long: -74.006,
  customer_lat: 34.0522,
  customer_long: -118.2437,
  dob: "1990-05-14",
  trans_time: "2026-08-29T12:00",
};

type FieldState = Record<keyof TransactionPayload, string>;

const EMPTY_FIELDS: FieldState = {
  amount: "",
  merchant: "",
  category: "",
  merchant_lat: "",
  merchant_long: "",
  customer_lat: "",
  customer_long: "",
  dob: "",
  trans_time: "",
};

function fieldsFromPayload(payload: TransactionPayload): FieldState {
  return {
    amount: String(payload.amount),
    merchant: payload.merchant,
    category: payload.category,
    merchant_lat: String(payload.merchant_lat),
    merchant_long: String(payload.merchant_long),
    customer_lat: String(payload.customer_lat),
    customer_long: String(payload.customer_long),
    dob: payload.dob,
    trans_time: payload.trans_time,
  };
}

interface FieldErrors {
  [key: string]: string | undefined;
}

function validate(fields: FieldState): { payload: TransactionPayload | null; errors: FieldErrors } {
  const errors: FieldErrors = {};

  const amount = Number(fields.amount);
  if (!fields.amount.trim() || Number.isNaN(amount) || amount <= 0) {
    errors.amount = "Enter an amount greater than 0.";
  }

  if (!fields.merchant.trim()) {
    errors.merchant = "Merchant is required.";
  }

  if (!fields.category.trim()) {
    errors.category = "Select a category.";
  }

  const coordFields: Array<[keyof FieldState, number, number]> = [
    ["merchant_lat", -90, 90],
    ["customer_lat", -90, 90],
    ["merchant_long", -180, 180],
    ["customer_long", -180, 180],
  ];
  const coords: Partial<Record<keyof FieldState, number>> = {};
  for (const [key, min, max] of coordFields) {
    const raw = fields[key];
    const num = Number(raw);
    if (!raw.trim() || Number.isNaN(num) || num < min || num > max) {
      errors[key] = `Enter a value between ${min} and ${max}.`;
    } else {
      coords[key] = num;
    }
  }

  if (!fields.dob.trim()) {
    errors.dob = "Date of birth is required.";
  }

  if (!fields.trans_time.trim()) {
    errors.trans_time = "Transaction time is required.";
  }

  if (Object.keys(errors).length > 0) {
    return { payload: null, errors };
  }

  return {
    payload: {
      amount,
      merchant: fields.merchant.trim(),
      category: fields.category,
      merchant_lat: coords.merchant_lat as number,
      merchant_long: coords.merchant_long as number,
      customer_lat: coords.customer_lat as number,
      customer_long: coords.customer_long as number,
      dob: fields.dob,
      trans_time: fields.trans_time.length === 16 ? `${fields.trans_time}:00` : fields.trans_time,
    },
    errors: {},
  };
}

interface TransactionFormProps {
  onSubmit: (payload: TransactionPayload) => void;
  isSubmitting: boolean;
}

export function TransactionForm({ onSubmit, isSubmitting }: TransactionFormProps) {
  const [devMode, setDevMode] = useState(false);
  const [fields, setFields] = useState<FieldState>(EMPTY_FIELDS);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [jsonPayload, setJsonPayload] = useState<TransactionPayload | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);

  function updateField<K extends keyof FieldState>(key: K, value: string) {
    setFields((prev) => ({ ...prev, [key]: value }));
  }

  function loadSample() {
    setFields(fieldsFromPayload(SAMPLE_PAYLOAD));
    setJsonPayload(SAMPLE_PAYLOAD);
    setErrors({});
    setJsonError(null);
  }

  function handleGuidedSubmit(event: FormEvent) {
    event.preventDefault();
    const { payload, errors: nextErrors } = validate(fields);
    setErrors(nextErrors);
    if (payload) onSubmit(payload);
  }

  function handleJsonSubmit(event: FormEvent) {
    event.preventDefault();
    if (jsonError) return;
    if (jsonPayload) onSubmit(jsonPayload);
  }

  return (
    <div className="rounded-lg border border-hairline bg-surface shadow-panel">
      <div className="flex items-center justify-between gap-3 border-b border-hairline px-5 py-4">
        <div>
          <h2 className="font-display text-base font-medium text-ink">Transaction details</h2>
          <p className="mt-0.5 text-xs text-ink-faint">
            Submitted to <span className="font-mono">/v1/predict-fraud</span>
          </p>
        </div>
        <label className="flex items-center gap-2.5">
          <span className="flex items-center gap-1.5 font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
            {devMode ? <Braces className="h-3.5 w-3.5" /> : <SlidersHorizontal className="h-3.5 w-3.5" />}
            {devMode ? "Raw JSON" : "Guided"}
          </span>
          <Switch checked={devMode} onCheckedChange={setDevMode} aria-label="Toggle raw JSON editor" />
        </label>
      </div>

      {devMode ? (
        <form onSubmit={handleJsonSubmit} className="p-5">
          <RawJsonEditor
            initialPayload={jsonPayload ?? (validate(fields).payload || undefined)}
            onChange={(payload, error) => {
              setJsonPayload(payload);
              setJsonError(error);
            }}
          />
          <FormFooter
            isSubmitting={isSubmitting}
            disabled={!jsonPayload || Boolean(jsonError)}
            onLoadSample={loadSample}
          />
        </form>
      ) : (
        <form onSubmit={handleGuidedSubmit} className="space-y-6 p-5">
          <section>
            <SectionLabel icon={<ListTree className="h-3.5 w-3.5" />} title="Transaction" />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <FieldGroup label="Amount (USD)" error={errors.amount} htmlFor="amount">
                <Input
                  id="amount"
                  inputMode="decimal"
                  placeholder="249.99"
                  value={fields.amount}
                  aria-invalid={Boolean(errors.amount)}
                  onChange={(e) => updateField("amount", e.target.value)}
                />
              </FieldGroup>
              <FieldGroup label="Category" error={errors.category} htmlFor="category">
                <Select value={fields.category} onValueChange={(v) => updateField("category", v)}>
                  <SelectTrigger id="category" aria-invalid={Boolean(errors.category)}>
                    <SelectValue placeholder="Select a category" />
                  </SelectTrigger>
                  <SelectContent>
                    {CATEGORIES.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldGroup>
              <FieldGroup label="Merchant" error={errors.merchant} htmlFor="merchant" className="sm:col-span-2">
                <Input
                  id="merchant"
                  placeholder="fraud_Kutch_and_Sons"
                  value={fields.merchant}
                  aria-invalid={Boolean(errors.merchant)}
                  onChange={(e) => updateField("merchant", e.target.value)}
                />
              </FieldGroup>
            </div>
          </section>

          <Separator />

          <section>
            <SectionLabel title="Merchant location" />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <FieldGroup label="Latitude" error={errors.merchant_lat} htmlFor="merchant_lat">
                <Input
                  id="merchant_lat"
                  inputMode="decimal"
                  placeholder="40.7128"
                  value={fields.merchant_lat}
                  aria-invalid={Boolean(errors.merchant_lat)}
                  onChange={(e) => updateField("merchant_lat", e.target.value)}
                />
              </FieldGroup>
              <FieldGroup label="Longitude" error={errors.merchant_long} htmlFor="merchant_long">
                <Input
                  id="merchant_long"
                  inputMode="decimal"
                  placeholder="-74.0060"
                  value={fields.merchant_long}
                  aria-invalid={Boolean(errors.merchant_long)}
                  onChange={(e) => updateField("merchant_long", e.target.value)}
                />
              </FieldGroup>
            </div>
          </section>

          <section>
            <SectionLabel title="Customer location" />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <FieldGroup label="Latitude" error={errors.customer_lat} htmlFor="customer_lat">
                <Input
                  id="customer_lat"
                  inputMode="decimal"
                  placeholder="34.0522"
                  value={fields.customer_lat}
                  aria-invalid={Boolean(errors.customer_lat)}
                  onChange={(e) => updateField("customer_lat", e.target.value)}
                />
              </FieldGroup>
              <FieldGroup label="Longitude" error={errors.customer_long} htmlFor="customer_long">
                <Input
                  id="customer_long"
                  inputMode="decimal"
                  placeholder="-118.2437"
                  value={fields.customer_long}
                  aria-invalid={Boolean(errors.customer_long)}
                  onChange={(e) => updateField("customer_long", e.target.value)}
                />
              </FieldGroup>
            </div>
          </section>

          <Separator />

          <section>
            <SectionLabel title="Identity & timing" />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <FieldGroup label="Customer date of birth" error={errors.dob} htmlFor="dob">
                <Input
                  id="dob"
                  type="date"
                  value={fields.dob}
                  aria-invalid={Boolean(errors.dob)}
                  onChange={(e) => updateField("dob", e.target.value)}
                />
              </FieldGroup>
              <FieldGroup label="Transaction timestamp" error={errors.trans_time} htmlFor="trans_time">
                <Input
                  id="trans_time"
                  type="datetime-local"
                  value={fields.trans_time}
                  aria-invalid={Boolean(errors.trans_time)}
                  onChange={(e) => updateField("trans_time", e.target.value)}
                />
              </FieldGroup>
            </div>
          </section>

          <FormFooter isSubmitting={isSubmitting} disabled={false} onLoadSample={loadSample} />
        </form>
      )}
    </div>
  );
}

function SectionLabel({ title, icon }: { title: string; icon?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-1.5 font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
      {icon}
      {title}
    </div>
  );
}

function FieldGroup({
  label,
  htmlFor,
  error,
  children,
  className,
}: {
  label: string;
  htmlFor: string;
  error?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="mb-1.5 flex items-baseline justify-between">
        <Label htmlFor={htmlFor}>{label}</Label>
        {error ? <span className="text-[0.68rem] text-signal-block">{error}</span> : null}
      </div>
      {children}
    </div>
  );
}

function FormFooter({
  isSubmitting,
  disabled,
  onLoadSample,
}: {
  isSubmitting: boolean;
  disabled: boolean;
  onLoadSample: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-hairline pt-5">
      <Button type="button" variant="ghost" size="sm" onClick={onLoadSample}>
        Load sample transaction
      </Button>
      <Button type="submit" size="lg" disabled={isSubmitting || disabled} className="min-w-[11rem]">
        {isSubmitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Scoring…
          </>
        ) : (
          "Score transaction"
        )}
      </Button>
    </div>
  );
}
