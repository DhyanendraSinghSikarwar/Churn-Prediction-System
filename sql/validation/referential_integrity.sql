-- Orphan-row checks across every FK in the schema. All should return zero
-- rows. FK constraints already prevent *new* orphans; this catches any
-- that could still arise out-of-band (a manual DELETE that bypassed the
-- FK via a superuser role, a partial restore from backup, etc.) and
-- serves as an explicit, reviewable statement of every FK relationship
-- in the schema in one place.

SELECT be.event_id FROM billing_events be
LEFT JOIN customers c ON c.customer_id = be.customer_id
WHERE c.customer_id IS NULL;

SELECT be.event_id FROM billing_events be
LEFT JOIN subscriptions s ON s.subscription_id = be.subscription_id
WHERE s.subscription_id IS NULL;

SELECT st.ticket_id FROM support_tickets st
LEFT JOIN customers c ON c.customer_id = st.customer_id
WHERE c.customer_id IS NULL;

SELECT s.subscription_id FROM subscriptions s
LEFT JOIN customers c ON c.customer_id = s.customer_id
WHERE c.customer_id IS NULL;

SELECT cl.customer_id FROM churn_labels cl
LEFT JOIN customers c ON c.customer_id = cl.customer_id
WHERE c.customer_id IS NULL;

SELECT fs.snapshot_id FROM feature_snapshots fs
LEFT JOIN customers c ON c.customer_id = fs.customer_id
WHERE c.customer_id IS NULL;

SELECT p.prediction_id FROM predictions p
LEFT JOIN customers c ON c.customer_id = p.customer_id
WHERE c.customer_id IS NULL;
