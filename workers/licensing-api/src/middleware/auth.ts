import type { MiddlewareHandler } from "hono";
import type { Env, JwtClaims } from "../types";
import { isJtiBlacklisted, verifyJwt } from "../lib/jwt";

export const authMiddleware: MiddlewareHandler<{
  Bindings: Env;
  Variables: { claims: JwtClaims };
}> = async (c, next) => {
  const header = c.req.header("Authorization");
  if (!header?.startsWith("Bearer ")) {
    return c.json({ error: "missing_token" }, 401);
  }

  const token = header.slice(7);
  const claims = await verifyJwt(token, c.env);
  if (!claims) {
    return c.json({ error: "invalid_token" }, 401);
  }

  const blacklisted = await isJtiBlacklisted(claims.jti, c.env.LICENSE_KV);
  if (blacklisted) {
    return c.json({ error: "token_revoked" }, 401);
  }

  c.set("claims", claims);
  await next();
};
