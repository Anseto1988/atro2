import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env } from "./types";
import { license } from "./routes/license";
import { webhook } from "./routes/webhook";
import { models } from "./routes/models";

const app = new Hono<{ Bindings: Env }>();

app.use("/*", (c, next) => {
  const origin = c.env.ALLOWED_ORIGIN ?? "https://astroai.app";
  return cors({ origin, allowMethods: ["GET", "POST", "OPTIONS"] })(c, next);
});

app.get("/health", (c) => c.json({ status: "ok", version: "0.1.0" }));

app.route("/api/v1/license", license);
app.route("/api/v1/webhook", webhook);
app.route("/api/v1/models", models);

app.notFound((c) => c.json({ error: "not_found" }, 404));

app.onError((err, c) => {
  console.error("Unhandled error:", err);
  return c.json({ error: "internal_error" }, 500);
});

export default app;
