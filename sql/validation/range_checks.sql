-- Range / domain checks. These duplicate what CHECK constraints already
-- enforce at write time — this file exists so the same invariants can be
-- checked against a bulk-loaded dataframe *before* it hits COPY (the
-- Phase 5 training pipeline's validation gate wires this in via Pandera,
-- see src/churn/features/validation.py), where a CHECK-constraint
-- violation would otherwise abort the whole load transaction instead of
-- failing early with a clear, row-level message. All should return zero
-- rows on a healthy database.

SELECT * FROM subscriptions WHERE monthly_charges < 0;

SELECT * FROM billing_events WHERE amount < 0;

SELECT * FROM support_tickets
WHERE satisfaction_score IS NOT NULL AND satisfaction_score NOT BETWEEN 0 AND 5;

SELECT * FROM support_tickets WHERE resolved_date IS NOT NULL AND resolved_date < created_date;

SELECT * FROM feature_snapshots WHERE tenure_months < 0;

SELECT * FROM feature_snapshots
WHERE tenure_bucket NOT IN ('0-6', '6-12', '12-24', '24-48', '48+');

SELECT * FROM feature_snapshots
WHERE payment_failure_rate_180d IS NOT NULL
  AND payment_failure_rate_180d NOT BETWEEN 0 AND 1;

SELECT * FROM feature_snapshots
WHERE frequency_charges_30d < 0 OR frequency_tickets_30d < 0 OR payment_failure_streak < 0;

SELECT * FROM predictions WHERE probability < 0 OR probability > 1;

SELECT * FROM predictions WHERE risk_band NOT IN ('low', 'medium', 'high');
