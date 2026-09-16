import type { ChangeEvent } from "react";

import {
  containsHan,
  humanizeIdentifier,
  localizedDynamicDescription,
  useI18n,
  type AppLocale,
} from "./i18n";

export type JsonSchema = {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  additionalProperties?: boolean | JsonSchema;
  format?: string;
  "x-dronedream-i18n"?: Partial<Record<AppLocale, {
    title?: string;
    description?: string;
  }>>;
};

export type CredentialReferenceOption = {
  reference: string;
  display_name: string;
};

export type SchemaField = {
  path: string[];
  key: string;
  label: string;
  required: boolean;
  schema: JsonSchema;
};

const LABELS: Record<string, { zh: string; en: string }> = {
  credential_reference: { zh: "凭证引用", en: "Credential reference" },
  database_id: { zh: "数据库 ID", en: "Database ID" },
  distance_weight: { zh: "距离权重", en: "Distance weight" },
  clearance_weight: { zh: "净空权重", en: "Clearance weight" },
  energy_weight: { zh: "能耗权重", en: "Energy weight" },
  stability_weight: { zh: "稳定性权重", en: "Stability weight" },
  qualification_weight: { zh: "资格状态权重", en: "Qualification weight" },
  reserve_fraction: { zh: "安全余量比例", en: "Reserve fraction" },
  qualified_range_m: { zh: "合格航程（米）", en: "Qualified range (m)" },
  minimum_turn_angle_deg: { zh: "最小转弯角（度）", en: "Minimum turn angle (deg)" },
  corner_speed_limit_mps: { zh: "转弯速度上限（米/秒）", en: "Corner speed limit (m/s)" },
  phase_speed_caps_mps: { zh: "分阶段速度上限（米/秒）", en: "Phase speed limits (m/s)" },
  maximum_join_distance_m: { zh: "最大衔接距离（米）", en: "Maximum join distance (m)" },
  headless: { zh: "无界面运行", en: "Run headless" },
  enabled: { zh: "启用", en: "Enabled" },
};

function labelFor(key: string, schema: JsonSchema): string {
  return schema.title || LABELS[key]?.zh || key.replaceAll("_", " ");
}

export function localizedSchemaFieldLabel(field: SchemaField, locale: AppLocale): string {
  const override = field.schema["x-dronedream-i18n"]?.[locale]?.title?.trim();
  if (override) return override;
  const known = LABELS[field.key];
  if (known) return locale === "en-US" ? known.en : known.zh;
  if (locale === "en-US") return containsHan(field.label) ? humanizeIdentifier(field.key) : field.label;
  return containsHan(field.label) ? field.label : `配置项（${humanizeIdentifier(field.key)}）`;
}

function localizedSchemaDescription(schema: JsonSchema, locale: AppLocale): string | undefined {
  const override = schema["x-dronedream-i18n"]?.[locale]?.description?.trim();
  return override || localizedDynamicDescription(schema.description, locale);
}

export function fieldsFromSchema(schema: JsonSchema, path: string[] = []): SchemaField[] {
  if (schema.type !== "object" || !schema.properties) return [];
  const required = new Set(schema.required ?? []);
  return Object.entries(schema.properties).map(([key, child]) => ({
    path: [...path, key],
    key,
    label: labelFor(key, child),
    required: required.has(key),
    schema: child,
  }));
}

export function valueAtPath(value: Record<string, unknown>, path: string[]): unknown {
  let current: unknown = value;
  for (const key of path) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

export function updateAtPath(
  value: Record<string, unknown>,
  path: string[],
  nextValue: unknown,
): Record<string, unknown> {
  if (path.length === 0) return value;
  const result = structuredClone(value);
  let current = result;
  for (const key of path.slice(0, -1)) {
    const existing = current[key];
    if (!existing || typeof existing !== "object" || Array.isArray(existing)) current[key] = {};
    current = current[key] as Record<string, unknown>;
  }
  const finalKey = path[path.length - 1];
  if (nextValue === undefined || nextValue === "") delete current[finalKey];
  else current[finalKey] = nextValue;
  return result;
}

function scalarType(schema: JsonSchema): string {
  const type = Array.isArray(schema.type)
    ? schema.type.find((item) => item !== "null")
    : schema.type;
  return type ?? (schema.enum ? "string" : "string");
}

function Field({ field, value, onChange, credentialReferences }: { field: SchemaField; value: Record<string, unknown>; onChange: (next: Record<string, unknown>) => void; credentialReferences: CredentialReferenceOption[] }) {
  const { locale, tr } = useI18n();
  const label = localizedSchemaFieldLabel(field, locale);
  const description = localizedSchemaDescription(field.schema, locale);
  const current = valueAtPath(value, field.path);
  const set = (next: unknown) => onChange(updateAtPath(value, field.path, next));
  const type = scalarType(field.schema);
  if (type === "object" && field.schema.properties) {
    return <fieldset className="schema-fieldset"><legend>{label}</legend><SchemaFields schema={field.schema} path={field.path} value={value} onChange={onChange} credentialReferences={credentialReferences} /></fieldset>;
  }
  if (field.schema.format === "dronedream-credential-reference") {
    return <label className="schema-field"><span>{label}{field.required && <b>{tr("必填", "Required")}</b>}</span><select value={typeof current === "string" ? current : ""} onChange={(event) => set(event.target.value)} required={field.required}><option value="">{tr("请选择安全凭证", "Select a secure credential")}</option>{credentialReferences.map((item) => <option key={item.reference} value={item.reference}>{item.display_name} · {item.reference.slice(-8)}</option>)}</select>{credentialReferences.length === 0 && <small>{tr("先在插件页顶部创建并授权一个凭证引用。", "Create and authorize a credential reference at the top of the Plugins page first.")}</small>}{description && <small>{description}</small>}</label>;
  }
  if (field.schema.enum) {
    return <label className="schema-field"><span>{label}{field.required && <b>{tr("必填", "Required")}</b>}</span><select value={typeof current === "string" ? current : ""} onChange={(event) => set(event.target.value)} required={field.required}><option value="">{tr("请选择", "Select")}</option>{field.schema.enum.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}</select>{description && <small>{description}</small>}</label>;
  }
  if (type === "boolean") {
    return <label className="schema-field schema-boolean"><span>{label}</span><button type="button" role="switch" aria-checked={current === true} className={`switch ${current === true ? "on" : ""}`} onClick={() => set(current !== true)}><span /></button>{description && <small>{description}</small>}</label>;
  }
  if (type === "array") {
    const text = Array.isArray(current) ? current.join("\n") : "";
    return <label className="schema-field"><span>{label}{field.required && <b>{tr("必填", "Required")}</b>}</span><textarea value={text} onChange={(event) => set(event.target.value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean))} placeholder={tr("每行一项", "One item per line")} required={field.required} />{description && <small>{description}</small>}</label>;
  }
  const numeric = type === "number" || type === "integer";
  const change = (event: ChangeEvent<HTMLInputElement>) => {
    if (!numeric) return set(event.target.value);
    if (!event.target.value) return set(undefined);
    const parsed = type === "integer" ? Number.parseInt(event.target.value, 10) : Number.parseFloat(event.target.value);
    set(Number.isFinite(parsed) ? parsed : undefined);
  };
  return <label className="schema-field"><span>{label}{field.required && <b>{tr("必填", "Required")}</b>}</span><input type={numeric ? "number" : "text"} value={typeof current === "string" || typeof current === "number" ? current : ""} onChange={change} required={field.required} min={field.schema.minimum} max={field.schema.maximum} minLength={field.schema.minLength} maxLength={field.schema.maxLength} autoComplete={field.key.includes("credential") ? "off" : undefined} />{description && <small>{description}</small>}</label>;
}

function SchemaFields({ schema, path, value, onChange, credentialReferences }: { schema: JsonSchema; path: string[]; value: Record<string, unknown>; onChange: (next: Record<string, unknown>) => void; credentialReferences: CredentialReferenceOption[] }) {
  return <div className="schema-fields">{fieldsFromSchema(schema, path).map((field) => <Field key={field.path.join(".")} field={field} value={value} onChange={onChange} credentialReferences={credentialReferences} />)}</div>;
}

export function JsonSchemaForm({ schema, value, onChange, credentialReferences = [] }: { schema: JsonSchema; value: Record<string, unknown>; onChange: (next: Record<string, unknown>) => void; credentialReferences?: CredentialReferenceOption[] }) {
  if (fieldsFromSchema(schema).length === 0) return null;
  return <SchemaFields schema={schema} path={[]} value={value} onChange={onChange} credentialReferences={credentialReferences} />;
}
