import { Hono } from "hono";
import type { Env, ActivateRequest, JwtClaims } from "../types";
import { getLicenseRecord, putLicenseRecord, putMachineIndex, removeMachineIndex, getKeyByMachineId } from "../lib/kv";
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
    await putMachineIndex(body.machine_id, body.license_key, c.env.LICENSE_KV);
  }

  const [{ token, expiresAt }, attestation] = await Promise.all([
    signJwt(record, body.machine_id, c.env),
    signAttestation(record.email, body.machine_id, c.env),
  ]);
  return c.json({ token, expires_at: expiresAt, attestation });
});

license.post("/refresh", authMiddleware, async (c) => {
  const claims = c.get("claims");

  const licenseKey = await getKeyByMachineId(claims.machine_id, c.env.LICENSE_KV);
  if (!licenseKey) {
    return c.json({ error: "license_revoked" }, 410);
  }

  const foundRecord = await getLicenseRecord(licenseKey, c.env.LICENSE_KV);
  if (!foundRecord || foundRecord.email !== claims.sub || !foundRecord.machine_ids.includes(claims.machine_id)) {
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

  const licenseKey = await getKeyByMachineId(claims.machine_id, c.env.LICENSE_KV);
  if (!licenseKey) {
    return c.json({ error: "license_not_found" }, 404);
  }

  const rec = await getLicenseRecord(licenseKey, c.env.LICENSE_KV);
  if (!rec || rec.email !== claims.sub || !rec.machine_ids.includes(claims.machine_id)) {
    return c.json({ error: "license_not_found" }, 404);
  }

  rec.machine_ids = rec.machine_ids.filter((id: string) => id !== claims.machine_id);
  rec.seats_used = rec.machine_ids.length;
  await putLicenseRecord(licenseKey, rec, c.env.LICENSE_KV);
  await removeMachineIndex(claims.machine_id, c.env.LICENSE_KV);
  await blacklistJti(claims.jti, claims.exp, c.env.LICENSE_KV);

  return c.json({ seats_released: 1 });
});

export { license };
