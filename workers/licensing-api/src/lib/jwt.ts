import type { AttestationClaims, Env, JwtClaims, LicenseRecord, LicenseTier } from "../types";
import { TIER_PLUGINS } from "../types";

const TOKEN_LIFETIME_SECONDS = 30 * 24 * 60 * 60; // 30 days

function pemToArrayBuffer(pem: string): ArrayBuffer {
  const lines = pem
    .replace(/-----BEGIN [\w\s]+-----/, "")
    .replace(/-----END [\w\s]+-----/, "")
    .replace(/\s/g, "");
  const binary = atob(lines);
  const buf = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    buf[i] = binary.charCodeAt(i);
  }
  return buf.buffer;
}

async function importPrivateKey(pem: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "pkcs8",
    pemToArrayBuffer(pem),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

async function importPublicKey(pem: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "spki",
    pemToArrayBuffer(pem),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"],
  );
}

function base64url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let str = "";
  for (const b of bytes) str += String.fromCharCode(b);
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecode(str: string): ArrayBuffer {
  const padded = str.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded);
  const buf = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    buf[i] = binary.charCodeAt(i);
  }
  return buf.buffer;
}

export async function signJwt(
  record: LicenseRecord,
  machineId: string,
  env: Env,
): Promise<{ token: string; expiresAt: string }> {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + TOKEN_LIFETIME_SECONDS;
  const jti = crypto.randomUUID();

  const claims: JwtClaims = {
    sub: record.email,
    jti,
    iat: now,
    exp,
    tier: record.tier,
    plugins: TIER_PLUGINS[record.tier] ?? [],
    machine_id: machineId,
    seats_used: record.seats_used,
    seats_max: record.seats_max,
  };

  const header = { alg: "RS256", typ: "JWT" };
  const enc = new TextEncoder();
  const headerB64 = base64url(enc.encode(JSON.stringify(header)));
  const payloadB64 = base64url(enc.encode(JSON.stringify(claims)));
  const signingInput = `${headerB64}.${payloadB64}`;

  const key = await importPrivateKey(env.JWT_PRIVATE_KEY);
  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    enc.encode(signingInput),
  );

  const token = `${signingInput}.${base64url(signature)}`;
  return { token, expiresAt: new Date(exp * 1000).toISOString() };
}

export async function signAttestation(
  userId: string,
  machineId: string,
  env: Env,
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const claims: AttestationClaims = {
    user_id: userId,
    machine_id: machineId,
    last_online_at: now,
    iat: now,
  };

  const header = { alg: "RS256", typ: "JWT" };
  const enc = new TextEncoder();
  const headerB64 = base64url(enc.encode(JSON.stringify(header)));
  const payloadB64 = base64url(enc.encode(JSON.stringify(claims)));
  const signingInput = `${headerB64}.${payloadB64}`;

  const key = await importPrivateKey(env.JWT_PRIVATE_KEY);
  const signature = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, enc.encode(signingInput));
  return `${signingInput}.${base64url(signature)}`;
}

export async function verifyAttestation(
  attestation: string,
  env: Env,
): Promise<AttestationClaims | null> {
  const parts = attestation.split(".");
  if (parts.length !== 3) return null;

  const [headerB64, payloadB64, signatureB64] = parts;
  const enc = new TextEncoder();
  try {
    const key = await importPublicKey(env.JWT_PUBLIC_KEY);
    const valid = await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      key,
      base64urlDecode(signatureB64),
      enc.encode(`${headerB64}.${payloadB64}`),
    );
    if (!valid) return null;
    return JSON.parse(new TextDecoder().decode(base64urlDecode(payloadB64))) as AttestationClaims;
  } catch {
    return null;
  }
}

export async function verifyJwt(
  token: string,
  env: Env,
): Promise<JwtClaims | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [headerB64, payloadB64, signatureB64] = parts;
  const enc = new TextEncoder();
  const signingInput = `${headerB64}.${payloadB64}`;

  try {
    const key = await importPublicKey(env.JWT_PUBLIC_KEY);
    const valid = await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      key,
      base64urlDecode(signatureB64),
      enc.encode(signingInput),
    );
    if (!valid) return null;

    const payload = JSON.parse(
      new TextDecoder().decode(base64urlDecode(payloadB64)),
    ) as JwtClaims;

    const now = Math.floor(Date.now() / 1000);
    if (payload.exp < now) return null;

    return payload;
  } catch {
    return null;
  }
}

export async function isJtiBlacklisted(
  jti: string,
  kv: KVNamespace,
): Promise<boolean> {
  const val = await kv.get(`jti_blacklist:${jti}`);
  return val !== null;
}

export async function blacklistJti(
  jti: string,
  expTimestamp: number,
  kv: KVNamespace,
): Promise<void> {
  const ttl = Math.max(expTimestamp - Math.floor(Date.now() / 1000), 60);
  await kv.put(`jti_blacklist:${jti}`, "1", { expirationTtl: ttl });
}
