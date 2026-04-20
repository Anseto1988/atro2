import { Hono } from "hono";
import type { Env, ActivateRequest, JwtClaims } from "../types";
import { getLicenseRecord, putLicenseRecord } from "../lib/kv";
import { signJwt, signAttestation, blacklistJti } from "../lib/jwt";
import { rateLimitMiddleware } from "../middleware/rate-limit";
import { authMiddleware } from "../middleware/auth";

const license = new Hono<{
  Bindings: Env;
  Variables: { claims: JwtClaims };
}>();

license.post("/activate", rateLimitMiddleware, async (c) => {
  const body = await c.req.json<ActivateRequest>();
  if (!body.license_key || !body.machine_id) {
    return c.json({ error: "invalid_request", message: "license_key and machine_id required" }, 400);
  }

  const record = await getLicenseRecord(body.license_key, c.env.LICENSE_KV);
  if (!record) {
    return c.json({ error: "invalid_key" }, 400);
  }

  const alreadyRegistered = record.machine_ids.includes(body.machine_id);

  if (!alreadyRegistered && record.seats_used >= record.seats_max) {
    return c.json({
      error: "max_seats_reached",
      seats_max: record.seats_max,
      seats_used: record.seats_used,
    }, 409);
  }

  if (!alreadyRegistered) {
    record.machine_ids.push(body.machine_id);
    record.seats_used = record.machine_ids.length;
    await putLicenseRecord(body.license_key, record, c.env.LICENSE_KV);
  }

  const [{ token, expiresAt }, attestation] = await Promise.all([
    signJwt(record, body.machine_id, c.env),
    signAttestation(record.email, body.machine_id, c.env),
  ]);
  return c.json({ token, expires_at: expiresAt, attestation });
});

license.post("/refresh", authMiddleware, async (c) => {
  const claims = c.get("claims");
  const licenseKeys = await c.env.LICENSE_KV.list({ prefix: "license:" });

  let foundRecord = null;
  let foundKey = "";
  for (const entry of licenseKeys.keys) {
    const raw = await c.env.LICENSE_KV.get(entry.name);
    if (!raw) continue;
    const rec = JSON.parse(raw);
    if (rec.email === claims.sub && rec.machine_ids?.includes(claims.machine_id)) {
      foundRecord = rec;
      foundKey = entry.name;
      break;
    }
  }

  if (!foundRecord) {
    return c.json({ error: "license_revoked" }, 410);
  }

  if (foundRecord.tier === "free" && claims.tier !== "free") {
    return c.json({ error: "subscription_expired" }, 403);
  }

  await blacklistJti(claims.jti, claims.exp, c.env.LICENSE_KV);
  const [{ token, expiresAt }, attestation] = await Promise.all([
    signJwt(foundRecord, claims.machine_id, c.env),
    signAttestation(foundRecord.email, claims.machine_id, c.env),
  ]);
  return c.json({ token, expires_at: expiresAt, attestation });
});

license.post("/deactivate", authMiddleware, async (c) => {
  const claims = c.get("claims");
  const licenseKeys = await c.env.LICENSE_KV.list({ prefix: "license:" });

  for (const entry of licenseKeys.keys) {
    const raw = await c.env.LICENSE_KV.get(entry.name);
    if (!raw) continue;
    const rec = JSON.parse(raw);
    if (rec.email === claims.sub && rec.machine_ids?.includes(claims.machine_id)) {
      rec.machine_ids = rec.machine_ids.filter((id: string) => id !== claims.machine_id);
      rec.seats_used = rec.machine_ids.length;
      await c.env.LICENSE_KV.put(entry.name, JSON.stringify(rec));

      await blacklistJti(claims.jti, claims.exp, c.env.LICENSE_KV);
      return c.json({ seats_released: 1 });
    }
  }

  return c.json({ error: "license_not_found" }, 404);
});

export { license };
