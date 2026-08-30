import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-[0.7rem] font-medium uppercase tracking-wide",
  {
    variants: {
      variant: {
        default: "border-hairline-strong bg-surface-raised text-ink-muted",
        approve: "border-signal-approve-border bg-signal-approve-bg text-signal-approve",
        review: "border-signal-review-border bg-signal-review-bg text-signal-review",
        block: "border-signal-block-border bg-signal-block-bg text-signal-block",
        wire: "border-wire-dim/40 bg-wire/10 text-wire",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
