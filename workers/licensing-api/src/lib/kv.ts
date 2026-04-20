import type { LicenseRecord } from "../types";

const KEY_PREFIX = "license:";
const SUB_INDEX_PREFIX = "sub_id:";
const MACHINE_INDEX_PREFIX = "machine:";

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

export async function putSubIndex(subId: string, licenseKey: string, kv: KVNamespace): Promise<void> {
  await kv.put(`${SUB_INDEX_PREFIX}${subId}`, licenseKey);
}

export async function putMachineIndex(machineId: string, licenseKey: string, kv: KVNamespace): Promise<void> {
  await kv.put(`${MACHINE_INDEX_PREFIX}${machineId}`, licenseKey);
}

export async function removeMachineIndex(machineId: string, kv: KVNamespace): Promise<void> {
  await kv.delete(`${MACHINE_INDEX_PREFIX}${machineId}`);
}

export async function getKeyByMachineId(machineId: string, kv: KVNamespace): Promise<string | null> {
  return kv.get(`${MACHINE_INDEX_PREFIX}${machineId}`);
}

export async function findLicenseByStripeSubId(
  subId: string,
  kv: KVNamespace,
): Promise<{ key: string; record: LicenseRecord } | null> {
  const licenseKey = await kv.get(`${SUB_INDEX_PREFIX}${subId}`);
  if (licenseKey) {
    const record = await getLicenseRecord(licenseKey, kv);
    if (record) return { key: licenseKey, record };
  }
  return null;
}

export function generateLicenseKey(): string {
  const seg = () => crypto.randomUUID().replace(/-/g, "").slice(0, 8).toUpperCase();
  return `ASTRO-${seg()}-${seg()}-${seg()}`;
}
