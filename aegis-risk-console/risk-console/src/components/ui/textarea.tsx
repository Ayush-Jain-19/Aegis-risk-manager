import * as React from "react";

import { cn } from "@/lib/utils";

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[220px] w-full rounded-md border border-hairline bg-surface-sunken px-3 py-2 text-sm text-ink",
          "font-mono leading-relaxed placeholder:text-ink-faint",
          "focus-visible:outline-none focus-visible:border-wire-dim focus-visible:ring-2 focus-visible:ring-wire/25",
          "aria-[invalid=true]:border-signal-block aria-[invalid=true]:focus-visible:ring-signal-block/25",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        ref={ref}
        spellCheck={false}
        {...props}
      />
    );
  },
);
Textarea.displayName = "Textarea";

export { Textarea };
