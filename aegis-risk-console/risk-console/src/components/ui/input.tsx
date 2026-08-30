import * as React from "react";

import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-hairline bg-surface-sunken px-3 py-2 text-sm text-ink",
          "placeholder:text-ink-faint",
          "transition-colors duration-150",
          "focus-visible:outline-none focus-visible:border-wire-dim focus-visible:ring-2 focus-visible:ring-wire/25",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "aria-[invalid=true]:border-signal-block aria-[invalid=true]:focus-visible:ring-signal-block/25",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

export { Input };
