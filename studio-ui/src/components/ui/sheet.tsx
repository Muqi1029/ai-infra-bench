import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "./button";

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;

interface SheetContentProps extends React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> {
  side?: "right" | "left";
}

export const SheetContent = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Content>,
  SheetContentProps
>(({ side = "right", className, children, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <DialogPrimitive.Overlay className="sheet-overlay fixed inset-0 z-40 bg-black/50 backdrop-blur-[1.5px] overscroll-contain" />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed inset-y-0 z-50 flex flex-col bg-[var(--surface)] text-[var(--text)] shadow-[var(--overlay-shadow)] overscroll-contain focus:outline-none",
        side === "right"
          ? "sheet-right right-0 w-full max-w-xl border-l border-[var(--border)]"
          : "sheet-left left-0 w-72 border-r border-[var(--rail-border)] bg-[var(--rail-bg)] text-[var(--rail-text)]",
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close
        className={cn(
          "absolute right-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-[7px] border border-transparent cursor-pointer transition-colors",
          side === "right"
            ? "text-[var(--text-muted)] hover:bg-[var(--surface-subtle)] hover:text-[var(--text)]"
            : "text-[var(--rail-muted)] hover:bg-[var(--rail-surface)] hover:text-[var(--rail-text)]"
        )}
      >
        <X className="h-4 w-4" strokeWidth={1.75} aria-hidden="true" />
        <span className="sr-only">关闭抽屉</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
));
SheetContent.displayName = "SheetContent";

export function SheetHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 py-4 border-b border-[var(--border)] pr-12", className)} {...props} />;
}

export const SheetTitle = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title ref={ref} className={cn("text-base font-bold text-[var(--text)] m-0", className)} {...props} />
));
SheetTitle.displayName = DialogPrimitive.Title.displayName;

export const SheetDescription = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn("text-xs text-[var(--text-muted)] mt-1 mb-0", className)}
    {...props}
  />
));
SheetDescription.displayName = DialogPrimitive.Description.displayName;

export interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel?: string;
  onConfirm: () => void | Promise<void>;
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "确认删除",
  onConfirm,
}: ConfirmDialogProps) {
  const [busy, setBusy] = React.useState(false);

  const handleConfirm = async () => {
    setBusy(true);
    try {
      await onConfirm();
      onOpenChange(false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="dialog-overlay fixed inset-0 z-50 bg-black/55 backdrop-blur-[1.5px] overscroll-contain" />
        <DialogPrimitive.Content className="dialog-content fixed left-1/2 top-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-[12px] border border-[var(--border)] bg-[var(--surface)] p-5 text-[var(--text)] shadow-[var(--overlay-shadow)] overscroll-contain">
          <DialogPrimitive.Title className="text-base font-bold m-0">{title}</DialogPrimitive.Title>
          <DialogPrimitive.Description className="text-[13px] text-[var(--text-muted)] mt-2 mb-5">
            {description}
          </DialogPrimitive.Description>
          <div className="flex justify-end gap-2">
            <DialogPrimitive.Close asChild>
              <Button variant="secondary" autoFocus>
                取消
              </Button>
            </DialogPrimitive.Close>
            <Button variant="danger-solid" loading={busy} onClick={handleConfirm}>
              {confirmLabel}
            </Button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: "neutral" | "brand" | "success" | "warning" | "danger";
}

export function Badge({ tone = "neutral", className, children, ...props }: BadgeProps) {
  const tones: Record<NonNullable<BadgeProps["tone"]>, string> = {
    neutral: "bg-[var(--surface-subtle)] text-[var(--text-sec)]",
    brand: "bg-[var(--brand-soft)] text-[var(--brand)]",
    success: "bg-[var(--emerald-soft)] text-[var(--emerald)]",
    warning: "bg-[var(--amber-soft)] text-[var(--amber)]",
    danger: "bg-[var(--crimson-soft)] text-[var(--crimson)]",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-[6px] px-2 py-0.5 text-[11.5px] font-semibold whitespace-nowrap",
        tones[tone],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
