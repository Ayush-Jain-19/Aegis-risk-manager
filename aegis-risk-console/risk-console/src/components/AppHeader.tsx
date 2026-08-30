import { useEffect, useRef, useState } from "react";
import { ShieldHalf } from "lucide-react";

import { checkApiConnectivity, API_BASE_URL, type ConnectivityStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 15_000;

function StatusPill({ status }: { status: ConnectivityStatus }) {
  const copy: Record<ConnectivityStatus, string> = {
    checking: "Checking",
    online: "API Connected",
    offline: "API Unreachable",
  };

  const dotClass: Record<ConnectivityStatus, string> = {
    checking: "bg-ink-faint",
    online: "bg-signal-approve",
    offline: "bg-signal-block",
  };

  return (
    <div
      className="flex items-center gap-2 rounded-sm border border-hairline bg-surface-raised px-2.5 py-1"
      role="status"
      aria-live="polite"
      title={API_BASE_URL}
    >
      <span className="relative flex h-1.5 w-1.5">
        <span
          className={cn(
            "absolute inline-flex h-full w-full rounded-full",
            dotClass[status],
            status === "checking" && "animate-pulse-dot",
          )}
        />
      </span>
      <span className="font-mono text-[0.7rem] uppercase tracking-wide text-ink-muted">
        {copy[status]}
      </span>
    </div>
  );
}

export function AppHeader() {
  const [status, setStatus] = useState<ConnectivityStatus>("checking");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    const controller = new AbortController();

    async function poll() {
      const result = await checkApiConnectivity(controller.signal);
      if (mountedRef.current) setStatus(result);
    }

    poll();
    const id = window.setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      mountedRef.current = false;
      controller.abort();
      window.clearInterval(id);
    };
  }, []);

  return (
    <header className="border-b border-hairline bg-surface/60 backdrop-blur">
      <div className="container flex h-16 max-w-7xl items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-hairline-strong bg-surface-raised text-wire">
            <ShieldHalf className="h-4 w-4" strokeWidth={1.75} />
          </span>
          <div className="flex items-baseline gap-2">
            <span className="font-display text-lg font-medium tracking-tight text-ink">
              Aegis
            </span>
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-faint">
              Risk Console
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden font-mono text-[0.7rem] uppercase tracking-wide text-ink-faint sm:inline">
            v1 · predict-fraud
          </span>
          <StatusPill status={status} />
        </div>
      </div>
    </header>
  );
}
