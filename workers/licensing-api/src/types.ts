import type { Context } from "hono";

export interface Env {
  LICENSE_KV: KVNamespace;
  MODELS_BUCKET: R2Bucket;
  JWT_PRIVATE_KEY: string;
  JWT_PUBLIC_KEY: string;
  STRIPE_SECRET_KEY: string;
  STRIPE_WEBHOOK_SECRET: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_ACCOUNT_ID: string;
  R2_BUCKET_NAME: string;
  ALLOWED_ORIGIN: string;
}

export type AppContext = Context<{ Bindings: Env; Variables: { claims?: JwtClaims } }>;

export type LicenseTier = "free" | "pro_monthly" | "pro_annual" | "founding_member";

export interface LicenseRecord {
  tier: LicenseTier;
  seats_max: number;
  seats_used: number;
  machine_ids: string[];
  stripe_sub_id: string | null;
  email: string;
  created_at: string;
}

export interface JwtClaims {
  sub: string;
  jti: string;
  iat: number;
  exp: number;
  tier: LicenseTier;
  plugins: string[];
  machine_id: string;
  seats_used: number;
  seats_max: number;
}

export interface AttestationClaims {
  user_id: string;
  machine_id: string;
  last_online_at: number; // Unix timestamp (server-authoritative)
  iat: number;
}

export interface ActivateRequest {
  license_key: string;
  machine_id: string;
  app_version: string;
}

export interface ModelManifestEntry {
  name: string;
  filename: string;
  size_bytes: number;
  sha256: string;
  description: string;
  min_tier: LicenseTier;
}

export interface ModelManifest {
  version: string;
  models: ModelManifestEntry[];
}

export const TIER_RANK: Record<LicenseTier, number> = {
  free: 0,
  pro_monthly: 1,
  pro_annual: 2,
  founding_member: 3,
};

export const TIER_PLUGINS: Record<LicenseTier, string[]> = {
  free: [],
  pro_monthly: ["denoise_pro", "starnet_pro", "stretch_ai"],
  pro_annual: ["denoise_pro", "starnet_pro", "stretch_ai"],
  founding_member: ["denoise_pro", "starnet_pro", "stretch_ai", "all_future"],
};
