import { Hono } from "hono";
import { AwsClient } from "aws4fetch";
import type { Env, JwtClaims, ModelManifest } from "../types";
import { TIER_RANK } from "../types";
import { authMiddleware } from "../middleware/auth";

const models = new Hono<{
  Bindings: Env;
  Variables: { claims: JwtClaims };
}>();

models.use("/*", authMiddleware);

const DEFAULT_MANIFEST: ModelManifest = {
  version: "0.1.0",
  models: [
    {
      name: "nafnet_denoise",
      filename: "nafnet_denoise.onnx",
      size_bytes: 0,
      sha256: "",
      description: "NAFNet denoising model (open-source, Apache 2.0)",
      min_tier: "free",
    },
    {
      name: "nafnet_denoise_pro",
      filename: "nafnet_denoise_pro.onnx",
      size_bytes: 0,
      sha256: "",
      description: "NAFNet Pro denoising model (proprietary, enhanced)",
      min_tier: "pro_monthly",
    },
    {
      name: "starnet_pro",
      filename: "starnet_pro.onnx",
      size_bytes: 0,
      sha256: "",
      description: "StarNet Pro star removal model (proprietary)",
      min_tier: "pro_monthly",
    },
    {
      name: "stretch_ai",
      filename: "stretch_ai.onnx",
      size_bytes: 0,
      sha256: "",
      description: "AI-powered histogram stretch model (proprietary)",
      min_tier: "pro_monthly",
    },
  ],
};

async function loadManifest(kv: KVNamespace): Promise<ModelManifest> {
  const raw = await kv.get("models:manifest");
  if (raw) return JSON.parse(raw) as ModelManifest;
  return DEFAULT_MANIFEST;
}

models.get("/manifest", async (c) => {
  const claims = c.get("claims");
  const userRank = TIER_RANK[claims.tier] ?? 0;
  const manifest = await loadManifest(c.env.LICENSE_KV);

  const filtered = manifest.models.filter(
    (m) => TIER_RANK[m.min_tier] <= userRank,
  );

  return c.json({ version: manifest.version, models: filtered });
});

models.post("/download-url", async (c) => {
  const claims = c.get("claims");
  const body = await c.req.json<{ model_name: string }>();
  if (!body.model_name) {
    return c.json({ error: "model_name_required" }, 400);
  }

  const manifest = await loadManifest(c.env.LICENSE_KV);
  const model = manifest.models.find((m) => m.name === body.model_name);
  if (!model) {
    return c.json({ error: "model_not_found" }, 404);
  }

  const userRank = TIER_RANK[claims.tier] ?? 0;
  if (TIER_RANK[model.min_tier] > userRank) {
    return c.json({ error: "tier_insufficient", required_tier: model.min_tier }, 403);
  }

  const r2 = new AwsClient({
    accessKeyId: c.env.R2_ACCESS_KEY_ID,
    secretAccessKey: c.env.R2_SECRET_ACCESS_KEY,
  });

  const bucketName = c.env.R2_BUCKET_NAME;
  const accountId = c.env.R2_ACCOUNT_ID;
  const objectKey = `models/${model.filename}`;
  const ttlSeconds = 3600;
  const expiresAt = new Date(Date.now() + ttlSeconds * 1000).toISOString();

  const url = new URL(
    `https://${bucketName}.${accountId}.r2.cloudflarestorage.com/${objectKey}`,
  );
  url.searchParams.set("X-Amz-Expires", String(ttlSeconds));

  const signed = await r2.sign(
    new Request(url.toString(), { method: "GET" }),
    { aws: { signQuery: true } },
  );

  return c.json({ url: signed.url, expires_at: expiresAt });
});

export { models };
