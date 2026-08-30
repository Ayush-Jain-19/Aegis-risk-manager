import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";

import { Textarea } from "@/components/ui/textarea";
import { parseTransactionPayload, type TransactionPayload } from "@/lib/types";

interface RawJsonEditorProps {
  initialPayload?: TransactionPayload;
  onChange: (payload: TransactionPayload | null, error: string | null) => void;
}

function stringify(payload: TransactionPayload | undefined): string {
  if (!payload) return "";
  return JSON.stringify(payload, null, 2);
}

export function RawJsonEditor({ initialPayload, onChange }: RawJsonEditorProps) {
  const [raw, setRaw] = useState(() => stringify(initialPayload));
  const [error, setError] = useState<string | null>(null);

  // Re-seed the editor when the parent hands us a new sample payload, but
  // don't fight the developer while they're actively typing their own JSON.
  useEffect(() => {
    if (initialPayload) setRaw(stringify(initialPayload));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPayload?.merchant, initialPayload?.trans_time]);

  useEffect(() => {
    const { payload, error: parseError } = parseTransactionPayload(raw);
    setError(parseError);
    onChange(payload, parseError);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw]);

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[0.68rem] uppercase tracking-wide text-ink-faint">
          Transaction payload (JSON)
        </span>
        {error ? (
          <span className="flex items-center gap-1 text-[0.68rem] text-signal-block">
            <AlertCircle className="h-3 w-3" /> {error}
          </span>
        ) : (
          <span className="flex items-center gap-1 text-[0.68rem] text-signal-approve">
            <CheckCircle2 className="h-3 w-3" /> Valid payload
          </span>
        )}
      </div>
      <Textarea
        value={raw}
        aria-invalid={Boolean(error)}
        onChange={(e) => setRaw(e.target.value)}
        placeholder='{\n  "amount": 249.99,\n  "merchant": "fraud_Kutch_and_Sons",\n  ...\n}'
      />
      <p className="mt-2 text-[0.72rem] text-ink-faint">
        Posted verbatim to <span className="font-mono">/v1/predict-fraud</span> — useful for
        replaying captured payloads or testing malformed input against the fallback engine.
      </p>
    </div>
  );
}
