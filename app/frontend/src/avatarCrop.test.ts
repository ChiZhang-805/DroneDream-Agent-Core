import { describe, expect, it } from "vitest";

import {
  AVATAR_MAX_ZOOM,
  avatarCropGeometry,
  clampAvatarCropOffset,
  clampAvatarCropZoom,
} from "./avatarCrop";

describe("avatar crop geometry", () => {
  it("fills a square crop without exposing gaps", () => {
    const wide = avatarCropGeometry({ width: 1200, height: 600 }, 300, 1);
    const tall = avatarCropGeometry({ width: 600, height: 1200 }, 300, 1);

    expect(wide.scale).toBe(0.5);
    expect(wide.maxOffsetX).toBe(150);
    expect(wide.maxOffsetY).toBe(0);
    expect(tall.maxOffsetX).toBe(0);
    expect(tall.maxOffsetY).toBe(150);
  });

  it("clamps drag and zoom so the circular crop always remains covered", () => {
    const geometry = avatarCropGeometry({ width: 1200, height: 600 }, 300, 2);

    expect(clampAvatarCropOffset({ x: 9_999, y: -9_999 }, geometry)).toEqual({
      x: geometry.maxOffsetX,
      y: -geometry.maxOffsetY,
    });
    expect(clampAvatarCropZoom(99)).toBe(AVATAR_MAX_ZOOM);
  });
});
