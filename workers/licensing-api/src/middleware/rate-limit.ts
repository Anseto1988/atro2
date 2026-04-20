import type { MiddlewareHandler } from "hono";
import type { Env } from "../types";

const MAX_ACTIVATIONS_PER_HOUR = 10;
const WINDOW_SECONDS = 3600;

interface RateEntry {
  count: number;
  window_start: number;
}

export const rateLimitMiddleware: MiddlewareHandler<{
  Bindings: Env;
}> = async (c, next) => {
  const ip = c.req.header("CF-Connecting-IP") ?? c.req.header("X-Forwarded-For") ?? "unknown";
  const key = `rate:activate:${ip}`;
  const now = Math.floor(Date.now() / 1000);

  const raw = await c.env.LICENSE_KV.get(key);
  let entry: RateEntry = raw ? JSON.parse(raw) : { count: 0, window_start: now };

  if (now - entry.window_start >= WINDOW_SECONDS) {
    entry = { count: 0, window_start: now };
  }

  if (entry.count >= MAX_ACTIVATIONS_PER_HOUR) {
    return c.json(
      { error: "rate_limited", retry_after: WINDOW_SECONDS - (now - entry.window_start) },
      429,
    );
  }

  entry.count++;
  const ttl = WINDOW_SECONDS - (now - entry.window_start);
  await c.env.LICENSE_KV.put(key, JSON.stringify(entry), { expirationTtl: Math.max(ttl, 60) });

  await next();
};
