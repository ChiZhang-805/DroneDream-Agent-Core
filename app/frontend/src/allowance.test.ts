import { describe, expect, it } from "vitest";

import { formatAllowanceRefillAt, remainingAllowancePercent } from "./allowance";

describe("remaining allowance", () => {
  it("renders a full bar at full allowance and shrinks as credits are consumed", () => {
    expect(remainingAllowancePercent(15_000_000, 15_000_000)).toBe(100);
    expect(remainingAllowancePercent(11_250_000, 15_000_000)).toBe(75);
    expect(remainingAllowancePercent(0, 15_000_000)).toBe(0);
  });

  it("clamps invalid server values and preserves the exact refill minute", () => {
    expect(remainingAllowancePercent(16_000_000, 15_000_000)).toBe(100);
    expect(remainingAllowancePercent(-1, 15_000_000)).toBe(0);
    expect(formatAllowanceRefillAt("not-a-date")).toBe("—");
    expect(formatAllowanceRefillAt("2026-07-31T08:23:00Z", "en-GB")).toContain("31/07/2026");
  });
});
