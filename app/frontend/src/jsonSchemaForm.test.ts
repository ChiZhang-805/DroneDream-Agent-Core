import { describe, expect, it } from "vitest";

import {
  fieldsFromSchema,
  localizedSchemaFieldLabel,
  updateAtPath,
  valueAtPath,
} from "./JsonSchemaForm";

describe("JSON Schema plugin configuration", () => {
  it("derives typed required fields without inventing configuration keys", () => {
    const fields = fieldsFromSchema({
      type: "object",
      required: ["credential_reference"],
      properties: {
        credential_reference: { type: "string" },
        reserve_fraction: { type: "number", minimum: 0.05, maximum: 0.8 },
        enabled: { type: "boolean" },
      },
    });
    expect(fields.map((field) => [field.key, field.required, field.schema.type])).toEqual([
      ["credential_reference", true, "string"],
      ["reserve_fraction", false, "number"],
      ["enabled", false, "boolean"],
    ]);
  });

  it("updates nested values immutably and removes an emptied optional value", () => {
    const original = { routing: { mode: "safe", retries: 2 } };
    const changed = updateAtPath(original, ["routing", "mode"], "rapid");
    const removed = updateAtPath(changed, ["routing", "retries"], undefined);
    expect(original.routing.mode).toBe("safe");
    expect(valueAtPath(changed, ["routing", "mode"])).toBe("rapid");
    expect(removed).toEqual({ routing: { mode: "rapid" } });
  });

  it("keeps built-in and extension schema labels in the selected language", () => {
    const builtIn = fieldsFromSchema({
      type: "object",
      properties: { maximum_join_distance_m: { type: "number" } },
    })[0];
    expect(localizedSchemaFieldLabel(builtIn, "zh-CN")).toBe("最大衔接距离（米）");
    expect(localizedSchemaFieldLabel(builtIn, "en-US")).toBe("Maximum join distance (m)");

    const extension = fieldsFromSchema({
      type: "object",
      properties: {
        vendor_mode: {
          type: "string",
          "x-dronedream-i18n": {
            "zh-CN": { title: "供应商模式" },
            "en-US": { title: "Provider mode" },
          },
        },
      },
    })[0];
    expect(localizedSchemaFieldLabel(extension, "zh-CN")).toBe("供应商模式");
    expect(localizedSchemaFieldLabel(extension, "en-US")).toBe("Provider mode");
  });
});
