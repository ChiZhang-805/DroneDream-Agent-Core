export function remainingAllowancePercent(remaining: number, included: number): number {
  if (!Number.isFinite(remaining) || !Number.isFinite(included) || included <= 0) return 0;
  return Math.max(0, Math.min(100, remaining / included * 100));
}

export function formatAllowanceRefillAt(value: string, locale = "zh-CN"): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}
