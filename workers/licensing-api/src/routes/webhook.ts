import { Hono } from "hono";
import Stripe from "stripe";
import type { Env, LicenseRecord, LicenseTier } from "../types";
import { findLicenseByStripeSubId, putLicenseRecord, putSubIndex, generateLicenseKey } from "../lib/kv";

const webhook = new Hono<{ Bindings: Env }>();

function stripePriceToTier(priceNickname: string | null, interval: string | null): LicenseTier {
  if (!priceNickname) return "pro_monthly";
  const lower = priceNickname.toLowerCase();
  if (lower.includes("founding")) return "founding_member";
  if (interval === "year") return "pro_annual";
  return "pro_monthly";
}

webhook.post("/stripe", async (c) => {
  const sigHeader = c.req.header("Stripe-Signature");
  if (!sigHeader) {
    return c.json({ error: "missing_signature" }, 400);
  }

  const stripe = new Stripe(c.env.STRIPE_SECRET_KEY, { apiVersion: "2026-03-25.dahlia" });
  const rawBody = await c.req.text();

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(rawBody, sigHeader, c.env.STRIPE_WEBHOOK_SECRET);
  } catch {
    return c.json({ error: "invalid_signature" }, 400);
  }
  const sub = event.data?.object as Record<string, any> | undefined;
  if (!sub) return c.json({ received: true });

  const subId: string = sub.id;
  let customerEmail: string = sub.metadata?.email ?? "";
  if (!customerEmail && sub.customer) {
    try {
      const customer = await stripe.customers.retrieve(sub.customer as string);
      if (!(customer as any).deleted) customerEmail = (customer as any).email ?? "";
    } catch {
      // customer lookup failed — proceed with empty email
    }
  }

  switch (event.type) {
    case "customer.subscription.created":
    case "customer.subscription.updated": {
      const status: string = sub.status;
      const priceNickname: string | null = sub.items?.data?.[0]?.price?.nickname ?? null;
      const interval: string | null = sub.items?.data?.[0]?.price?.recurring?.interval ?? null;

      const existing = await findLicenseByStripeSubId(subId, c.env.LICENSE_KV);

      if (existing) {
        const tier = status === "active" ? stripePriceToTier(priceNickname, interval) : "free";
        existing.record.tier = tier;
        await putLicenseRecord(existing.key, existing.record, c.env.LICENSE_KV);
      } else if (status === "active" && customerEmail) {
        const licenseKey = generateLicenseKey();
        const tier = stripePriceToTier(priceNickname, interval);
        const seatsMax = tier === "founding_member" ? 2 : 1;
        const record: LicenseRecord = {
          tier,
          seats_max: seatsMax,
          seats_used: 0,
          machine_ids: [],
          stripe_sub_id: subId,
          email: customerEmail,
          created_at: new Date().toISOString(),
        };
        await putLicenseRecord(licenseKey, record, c.env.LICENSE_KV);
        await putSubIndex(subId, licenseKey, c.env.LICENSE_KV);
      }
      break;
    }

    case "customer.subscription.deleted": {
      const existing = await findLicenseByStripeSubId(subId, c.env.LICENSE_KV);
      if (existing) {
        existing.record.tier = "free";
        await putLicenseRecord(existing.key, existing.record, c.env.LICENSE_KV);
      }
      break;
    }
  }

  return c.json({ received: true });
});

export { webhook };
