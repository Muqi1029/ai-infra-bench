import * as React from "react";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger" | "danger-solid";
  size?: "sm" | "md" | "lg" | "icon";
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", size = "md", loading = false, disabled, children, type = "button", ...props }, ref) => {
    const variants: Record<NonNullable<ButtonProps["variant"]>, string> = {
      primary:
        "bg-[var(--brand)] text-[var(--brand-fg)] border border-[var(--brand)] hover:opacity-92 active:opacity-85",
      secondary:
        "bg-[var(--control-bg)] text-[var(--text)] border border-[var(--control-border)] hover:border-[var(--control-border-hover)] hover:bg-[var(--surface-subtle)]",
      ghost:
        "bg-transparent text-[var(--text-sec)] border border-transparent hover:bg-[var(--surface-subtle)] hover:text-[var(--text)]",
      danger:
        "bg-[var(--control-bg)] text-[var(--crimson)] border border-[var(--control-border)] hover:border-[var(--crimson)] hover:bg-[var(--crimson-soft)]",
      "danger-solid":
        "bg-[var(--crimson)] text-white border border-[var(--crimson)] hover:opacity-92",
    };

    const sizes: Record<NonNullable<ButtonProps["size"]>, string> = {
      sm: "min-h-[32px] px-2.5 py-1 text-xs rounded-[7px] gap-1.5",
      md: "min-h-[38px] px-3.5 py-1.5 text-[13px] rounded-[8px] gap-1.5",
      lg: "min-h-[42px] px-4.5 py-2 text-sm rounded-[8px] gap-2",
      icon: "min-h-[34px] min-w-[34px] p-1.5 rounded-[8px]",
    };

    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled || loading}
        className={cn(
          "inline-flex items-center justify-center font-semibold whitespace-nowrap cursor-pointer select-none transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {loading && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" strokeWidth={1.75} aria-hidden="true" />}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";
