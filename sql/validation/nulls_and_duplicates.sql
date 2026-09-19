-- Null and duplicate-key checks. Every SELECT below should return zero
-- rows on a healthy database. Most of these are also enforced by schema
-- constraints (PK/UNIQUE/NOT NULL) — this file exists as an independent,
-- queryable audit (run ad hoc via `psql -f`, or wired into the Phase 5
-- training pipeline's validation gate) rather than trusting that nothing
-- ever bypassed the ORM/COPY path.

-- 1. Duplicate customers (should be structurally impossible: customer_id is PK).
SELECT customer_id, COUNT(*)
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;

-- 2. Duplicate feature_snapshots rows for the same panel cell.
SELECT customer_id, reference_date, feature_set_version, COUNT(*)
FROM feature_snapshots
GROUP BY customer_id, reference_date, feature_set_version
HAVING COUNT(*) > 1;

-- 3. feature_snapshots rows with a null in a column the schema declares
--    NOT NULL — belt-and-braces check for anyone loading via raw SQL
--    rather than the COPY path in build_panel.py.
SELECT snapshot_id
FROM feature_snapshots
WHERE tenure_months IS NULL
   OR tenure_bucket IS NULL
   OR frequency_charges_30d IS NULL
   OR frequency_tickets_30d IS NULL
   OR monetary_charges_30d IS NULL
   OR cumulative_charges_to_date IS NULL
   OR payment_failure_streak IS NULL;

-- 4. Every customer must have exactly one churn_labels row (set at
--    ingestion time — a missing row here means ingestion partially failed).
SELECT c.customer_id
FROM customers c
LEFT JOIN churn_labels cl ON cl.customer_id = c.customer_id
WHERE cl.customer_id IS NULL;

-- 5. Every customer must have exactly one *current* subscription
--    (end_date IS NULL) unless they have churned.
SELECT c.customer_id
FROM customers c
JOIN churn_labels cl ON cl.customer_id = c.customer_id AND cl.churned = false
LEFT JOIN subscriptions s ON s.customer_id = c.customer_id AND s.end_date IS NULL
WHERE s.subscription_id IS NULL;
