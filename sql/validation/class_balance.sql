-- Reports the monthly churn_next_30d positive rate per reference_date
-- cohort in feature_snapshots — the empirical proof of the imbalance
-- called out in PLAN.md and
-- DECISIONS.md#imbalance-is-designed-in. Expect a low single-digit
-- percentage per month, well below the 26.6% *cumulative* churn rate of
-- the raw dataset (that figure is churn-over-the-whole-relationship, not
-- a monthly rate — see DECISIONS.md for why those two numbers are not
-- the same thing and shouldn't be compared directly).

SELECT
    reference_date,
    COUNT(*) AS cohort_size,
    COUNT(*) FILTER (WHERE label_churn_next_30d) AS positives,
    ROUND(100.0 * COUNT(*) FILTER (WHERE label_churn_next_30d) / COUNT(*), 2) AS positive_rate_pct
FROM feature_snapshots
GROUP BY reference_date
ORDER BY reference_date;

-- Overall panel-wide rate, for a single headline number.
SELECT
    COUNT(*) AS panel_size,
    COUNT(*) FILTER (WHERE label_churn_next_30d) AS positives,
    ROUND(100.0 * COUNT(*) FILTER (WHERE label_churn_next_30d) / COUNT(*), 2) AS positive_rate_pct
FROM feature_snapshots;
