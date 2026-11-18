import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cn } from "../../lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type = "text", ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        "w-full min-h-[38px] px-3 py-1.5 rounded-[8px] text-[13.5px] bg-[var(--control-bg)] text-[var(--text)] border border-[var(--control-border)] placeholder:text-[var(--text-muted)] hover:border-[var(--control-border-hover)] transition-colors duration-150 disabled:opacity-55 disabled:cursor-not-allowed",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";

export const Label = React.forwardRef<
  React.ComponentRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root>
>(({ className, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    className={cn("text-[12.5px] font-semibold text-[var(--text-sec)] leading-snug select-none", className)}
    {...props}
  />
));
Label.displayName = "Label";
