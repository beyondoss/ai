/**
 * Eve-compatible schema handling: Zod / Standard Schema / plain JSON Schema.
 * We do not depend on a Zod major — whatever the agent installs speaks Standard Schema.
 */
import { createRequire } from "node:module";

import type { StandardSchemaV1 } from "@standard-schema/spec";

const require = createRequire(import.meta.url);

export type JsonSchema = Record<string, unknown>;

export type ToolInputSchema = StandardSchemaV1<unknown, unknown> | JsonSchema;

export type InferToolInput<S> = S extends StandardSchemaV1<infer _In, infer Out>
  ? Out
  : Record<string, unknown>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Standard Schema v1 (Zod 3.24+, Zod 4, Valibot, ArkType, …). */
export function isStandardSchema(
  schema: unknown,
): schema is StandardSchemaV1<unknown, unknown> {
  if (!isRecord(schema)) return false;
  const standard = schema["~standard"];
  return (
    isRecord(standard) &&
    standard.version === 1 &&
    typeof standard.validate === "function"
  );
}

/**
 * Plain JSON Schema object (not a Standard Schema instance).
 * Eve: JSON Schema reaches the model as written (minus `$schema`).
 */
export function isPlainJsonSchema(schema: unknown): schema is JsonSchema {
  if (!isRecord(schema) || isStandardSchema(schema)) return false;
  return (
    "type" in schema ||
    "properties" in schema ||
    "anyOf" in schema ||
    "oneOf" in schema ||
    "allOf" in schema ||
    "$ref" in schema
  );
}

export function stripDollarSchema(schema: JsonSchema): JsonSchema {
  const { $schema: _drop, ...rest } = schema;
  return rest;
}

/**
 * JSON Schema the model / MCP `tools/list` / `.mcp-tools.json` see.
 * Prefers Standard JSON Schema, then Zod 4 `toJSONSchema`, then plain JSON Schema.
 */
export function inputSchemaToJsonSchema(schema: ToolInputSchema): JsonSchema {
  if (isStandardSchema(schema)) {
    const props = schema["~standard"] as StandardSchemaV1.Props<unknown, unknown> & {
      jsonSchema?: {
        input?: (opts?: { target?: string }) => JsonSchema;
      };
    };
    const fromStandard = props.jsonSchema?.input?.({ target: "draft-2020-12" });
    if (fromStandard && isRecord(fromStandard)) {
      return stripDollarSchema(fromStandard);
    }
  }

  // Zod 4 instance method (when Standard JSON Schema trait is absent).
  if (
    isRecord(schema) &&
    typeof schema.toJSONSchema === "function"
  ) {
    const raw = (schema.toJSONSchema as (opts?: { target?: string }) => unknown)({
      target: "draft-2020-12",
    });
    if (isRecord(raw)) return stripDollarSchema(raw);
  }

  if (isPlainJsonSchema(schema)) {
    return stripDollarSchema({ ...schema });
  }

  // Zod 4 module helper (peer).
  try {
    const z = require("zod") as {
      toJSONSchema?: (s: unknown, opts?: { target?: string }) => JsonSchema;
    };
    if (typeof z.toJSONSchema === "function") {
      return stripDollarSchema(z.toJSONSchema(schema, { target: "draft-2020-12" }));
    }
  } catch {
    /* zod not installed at resolve time — agent may still pass Standard Schema */
  }

  // Zod 3 without Standard JSON Schema: optional zod-to-json-schema.
  try {
    const { zodToJsonSchema } = require("zod-to-json-schema") as {
      zodToJsonSchema: (s: unknown) => JsonSchema;
    };
    return stripDollarSchema(zodToJsonSchema(schema));
  } catch {
    /* optional */
  }

  throw new Error(
    "inputSchema must be a Zod/Standard Schema or a plain JSON Schema object (Eve-compatible). For Zod 3, install zod-to-json-schema; Zod 4 exports JSON Schema via Standard Schema / toJSONSchema.",
  );
}

export class ToolInputValidationError extends Error {
  readonly issues: readonly unknown[];
  constructor(message: string, issues: readonly unknown[] = []) {
    super(message);
    this.name = "ToolInputValidationError";
    this.issues = issues;
  }
}

/** Validate model args the way Eve does: library validate for Standard Schema, local JSON Schema otherwise. */
export async function validateToolInput<S extends ToolInputSchema>(
  schema: S,
  raw: unknown,
): Promise<InferToolInput<S>> {
  if (isStandardSchema(schema)) {
    let result = schema["~standard"].validate(raw);
    if (result instanceof Promise) result = await result;
    if (result.issues) {
      const msg = result.issues
        .map((i) => i.message)
        .filter(Boolean)
        .join("; ");
      throw new ToolInputValidationError(
        msg || "invalid tool input",
        result.issues,
      );
    }
    return result.value as InferToolInput<S>;
  }

  if (isPlainJsonSchema(schema)) {
    // Eve: fill omitted properties from `default`, then validate.
    const withDefaults = applyJsonSchemaDefaults(schema, raw);
    const checked = validatePlainJsonSchema(schema, withDefaults, []);
    if (checked !== undefined) {
      throw new ToolInputValidationError(checked);
    }
    return withDefaults as InferToolInput<S>;
  }

  throw new Error("inputSchema must be a Zod/Standard Schema or a plain JSON Schema object");
}

/** Eve: fill omitted properties from their JSON Schema `default` values (recursive). */
export function applyJsonSchemaDefaults(schema: JsonSchema, raw: unknown): unknown {
  if (!isRecord(schema.properties)) {
    return raw;
  }
  const base =
    raw !== null && typeof raw === "object" && !Array.isArray(raw)
      ? { ...(raw as Record<string, unknown>) }
      : {};

  for (const [key, propSchema] of Object.entries(schema.properties)) {
    if (!isRecord(propSchema)) continue;
    if (!(key in base) && "default" in propSchema) {
      base[key] = propSchema.default;
    } else if (key in base && isRecord(propSchema.properties)) {
      base[key] = applyJsonSchemaDefaults(propSchema, base[key]);
    }
  }
  return base;
}

/**
 * Pragmatic JSON Schema 2020-12 / draft-07 subset (type, properties, required, additionalProperties).
 * Unknown keywords are ignored (Eve: when unevaluable, pass through — we still try the common path).
 */
function validatePlainJsonSchema(
  schema: JsonSchema,
  value: unknown,
  path: (string | number)[],
): string | undefined {
  const loc = path.length ? path.join(".") : "(root)";

  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    const ok = types.some((t) => matchesJsonType(t, value));
    if (!ok) {
      return `${loc}: expected ${types.join("|")}, got ${describeType(value)}`;
    }
  }

  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    const props = isRecord(schema.properties) ? schema.properties : undefined;
    const required = Array.isArray(schema.required)
      ? schema.required.filter((k): k is string => typeof k === "string")
      : [];

    for (const key of required) {
      if (!(key in obj)) return `${loc}: missing required property "${key}"`;
    }

    if (props) {
      for (const [key, propSchema] of Object.entries(props)) {
        if (!(key in obj)) continue;
        if (!isRecord(propSchema)) continue;
        const err = validatePlainJsonSchema(propSchema, obj[key], [...path, key]);
        if (err) return err;
      }
    }

    if (schema.additionalProperties === false && props) {
      for (const key of Object.keys(obj)) {
        if (!(key in props)) return `${loc}: unexpected property "${key}"`;
      }
    }
  }

  if (Array.isArray(value) && isRecord(schema.items)) {
    for (let i = 0; i < value.length; i++) {
      const err = validatePlainJsonSchema(schema.items, value[i], [...path, i]);
      if (err) return err;
    }
  }

  return undefined;
}

function matchesJsonType(type: unknown, value: unknown): boolean {
  switch (type) {
    case "object":
      return value !== null && typeof value === "object" && !Array.isArray(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && !Number.isNaN(value);
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return true; // unknown type name → Eve: do not reject
  }
}

function describeType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}
