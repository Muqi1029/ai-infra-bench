import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data as T;
}

export function parseNumList(raw: string, isFloat = false): number[] {
  return String(raw || "")
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((v) => (isFloat ? parseFloat(v) : parseInt(v, 10)))
    .filter((v) => Number.isFinite(v) && v > 0);
}

export function fmtNum(val: unknown, digits = 2): string {
  return typeof val === "number" && Number.isFinite(val) ? val.toFixed(digits) : "—";
}
