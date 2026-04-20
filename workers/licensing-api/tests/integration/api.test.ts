import { describe, it, expect, beforeEach } from "vitest";
import app from "../../src/index";
import type { LicenseRecord } from "../../src/types";

const TEST_LICENSE_KEY = "ASTRO-TEST-1234-ABCD";
const TEST_MACHINE_ID = "sha256:test_machine_001";

const TEST_LICENSE: LicenseRecord = {
  tier: "pro_annual",
  seats_max: 1,
  seats_used: 0,
  machine_ids: [],
  stripe_sub_id: "sub_test_123",
  email: "astro@example.com",
  created_at: "2026-01-01T00:00:00Z",
};

class MockKVNamespace {
  private store = new Map<string, { value: string; expiration?: number }>();

  async get(key: string): Promise<string | null> {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (entry.expiration && Date.now() / 1000 > entry.expiration) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  async put(key: string, value: string, opts?: { expirationTtl?: number }): Promise<void> {
    const expiration = opts?.expirationTtl
      ? Math.floor(Date.now() / 1000) + opts.expirationTtl
      : undefined;
    this.store.set(key, { value, expiration });
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async list(opts?: { prefix?: string }): Promise<{ keys: { name: string }[] }> {
    const keys: { name: string }[] = [];
    for (const name of this.store.keys()) {
      if (!opts?.prefix || name.startsWith(opts.prefix)) {
        keys.push({ name });
      }
    }
    return { keys };
  }

  seed(key: string, value: string): void {
    this.store.set(key, { value });
  }

  clear(): void {
    this.store.clear();
  }
}

function createMockEnv(kv: MockKVNamespace) {
  return {
    LICENSE_KV: kv as unknown as KVNamespace,
    MODELS_BUCKET: {} as R2Bucket,
    JWT_PRIVATE_KEY: "MOCK_KEY",
    JWT_PUBLIC_KEY: "MOCK_KEY",
    STRIPE_SECRET_KEY: "sk_test_fake_key",
    STRIPE_WEBHOOK_SECRET: "whsec_test_secret",
    R2_ACCESS_KEY_ID: "test_key",
    R2_SECRET_ACCESS_KEY: "test_secret",
    R2_ACCOUNT_ID: "test_account",
    R2_BUCKET_NAME: "test-bucket",
  };
}

describe("Health endpoint", () => {
  it("returns ok", async () => {
    const kv = new MockKVNamespace();
    const env = createMockEnv(kv);
    const res = await app.request("/health", {}, env);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({ status: "ok", version: "0.1.0" });
  });
});

describe("POST /api/v1/license/activate", () => {
  let kv: MockKVNamespace;
  let env: ReturnType<typeof createMockEnv>;

  beforeEach(() => {
    kv = new MockKVNamespace();
    env = createMockEnv(kv);
    kv.seed(`license:${TEST_LICENSE_KEY}`, JSON.stringify(TEST_LICENSE));
  });

  it("returns 400 for missing license_key", async () => {
    const res = await app.request(
      "/api/v1/license/activate",
      { method: "POST", body: JSON.stringify({ machine_id: TEST_MACHINE_ID }), headers: { "Content-Type": "application/json" } },
      env,
    );
    expect(res.status).toBe(400);
  });

  it("returns 400 for invalid key", async () => {
    const res = await app.request(
      "/api/v1/license/activate",
      {
        method: "POST",
        body: JSON.stringify({ license_key: "INVALID", machine_id: TEST_MACHINE_ID, app_version: "0.1.0" }),
        headers: { "Content-Type": "application/json" },
      },
      env,
    );
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("invalid_key");
  });

  it("returns 409 when max seats reached", async () => {
    const fullLicense: LicenseRecord = {
      ...TEST_LICENSE,
      seats_used: 1,
      machine_ids: ["sha256:other_machine"],
    };
    kv.seed(`license:${TEST_LICENSE_KEY}`, JSON.stringify(fullLicense));

    const res = await app.request(
      "/api/v1/license/activate",
      {
        method: "POST",
        body: JSON.stringify({ license_key: TEST_LICENSE_KEY, machine_id: TEST_MACHINE_ID, app_version: "0.1.0" }),
        headers: { "Content-Type": "application/json" },
      },
      env,
    );
    expect(res.status).toBe(409);
    const body = await res.json();
    expect(body.error).toBe("max_seats_reached");
  });
});

describe("POST /api/v1/webhook/stripe", () => {
  it("returns 400 without Stripe-Signature", async () => {
    const kv = new MockKVNamespace();
    const env = createMockEnv(kv);
    const res = await app.request(
      "/api/v1/webhook/stripe",
      { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } },
      env,
    );
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("missing_signature");
  });
});

describe("GET /api/v1/models/manifest", () => {
  it("returns 401 without auth", async () => {
    const kv = new MockKVNamespace();
    const env = createMockEnv(kv);
    const res = await app.request("/api/v1/models/manifest", {}, env);
    expect(res.status).toBe(401);
  });
});

describe("POST /api/v1/models/download-url", () => {
  it("returns 401 without auth", async () => {
    const kv = new MockKVNamespace();
    const env = createMockEnv(kv);
    const res = await app.request(
      "/api/v1/models/download-url",
      { method: "POST", body: JSON.stringify({ model_name: "test" }), headers: { "Content-Type": "application/json" } },
      env,
    );
    expect(res.status).toBe(401);
  });
});

describe("404 handler", () => {
  it("returns not_found for unknown routes", async () => {
    const kv = new MockKVNamespace();
    const env = createMockEnv(kv);
    const res = await app.request("/unknown", {}, env);
    expect(res.status).toBe(404);
  });
});
