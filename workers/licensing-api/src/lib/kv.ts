import type { LicenseRecord } from "../types";

const KEY_PREFIX = "license:";

export async function getLicenseRecord(
  key: string,
  kv: KVNamespace,
): Promise<LicenseRecord | null> {
  const raw = await kv.get(`${KEY_PREFIX}${key}`);
  if (!raw) return null;
  return JSON.parse(raw) as LicenseRecord;
}

export async function putLicenseRecord(
  key: string,
  record: LicenseRecord,
  kv: KVNamespace,
): Promise<void> {
  await kv.put(`${KEY_PREFIX}${key}`, JSON.stringify(record));
}

export async function findLicenseByStripeSubId(
  subId: string,
  kv: KVNamespace,
): Promise<{ key: string; record: LicenseRecord } | null> {
  let cursor: string | undefined;
  do {
    const result = await kv.list({ prefix: KEY_PREFIX, cursor });
    for (const entry of result.keys) {
      const raw = await kv.get(entry.name);
      if (!raw) continue;
      const record = JSON.parse(raw) as LicenseRecord;
      if (record.stripe_sub_id === subId) {
        return { key: entry.name.slice(KEY_PREFIX.length), record };
      }
    }
    if (!result.list_complete) cursor = result.cursor;
    else break;
  } while (true);
  return null;
}
