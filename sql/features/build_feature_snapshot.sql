-- Point-in-time feature extraction for a single reference_date. This is
-- the ONE query that serves both training (called once per historical
-- monthly cohort — see src/churn/features/build_panel.py) and inference
-- (Phase 6: called once with reference_date = today) — see PLAN.md
-- Phase 0, "a reference_date parameter so the same query serves training
-- and inference."
--
-- POINT-IN-TIME CORRECTNESS: every feature CTE below filters strictly to
-- event_date < :reference_date (billing_events) or
-- created_date < :reference_date (support_tickets) — never <=. No
-- feature column can see an event dated on or after the cutoff. This is
-- checked independently of this query's own logic by
-- sql/validation/point_in_time_reconciliation.sql, which recomputes a
-- couple of the aggregates directly from source tables and diffs them.
--
-- LABEL LOOKAHEAD IS NOT LEAKAGE: label_churn_next_30d looks FORWARD from
-- reference_date to reference_date + :horizon_days — that is expected of
-- a supervised label, not a violation of the rule above, as long as it
-- never feeds a feature column (it doesn't — it's computed only in the
-- final SELECT, from cohort.churn_date, never joined into a feature
-- CTE). See DECISIONS.md#label-lookahead-is-not-leakage.
--
-- Params:
--   :reference_date        date  -- the feature cutoff / scoring date
--   :horizon_days           int  -- label window length (config: data.prediction_horizon_days)
--   :feature_set_version   text  -- tag written into every row (config: feature_set.version)

WITH cohort AS (
    -- Customers who were an active subscriber going into reference_date:
    -- already signed up, and either never churned or churned strictly
    -- after reference_date. This is the panel "spine" for this one
    -- reference_date.
    SELECT
        c.customer_id,
        c.signup_date,
        c.senior_citizen,
        c.partner,
        c.dependents,
        s.contract_type,
        s.internet_service,
        s.online_security,
        s.online_backup,
        s.device_protection,
        s.tech_support,
        s.streaming_tv,
        s.streaming_movies,
        s.paperless_billing,
        s.payment_method,
        s.monthly_charges,
        cl.churn_date
    FROM customers c
    JOIN subscriptions s
        ON s.customer_id = c.customer_id
        AND s.start_date <= :reference_date
        AND (s.end_date IS NULL OR s.end_date > :reference_date)
    JOIN churn_labels cl ON cl.customer_id = c.customer_id
    WHERE c.signup_date <= :reference_date
      AND (cl.churn_date IS NULL OR cl.churn_date > :reference_date)
),

-- ---------------------------------------------------------------------
-- Billing-derived features. billing_history is the single point-in-time
-- filter every billing feature below draws from.
-- ---------------------------------------------------------------------
billing_history AS (
    SELECT be.customer_id, be.event_date, be.event_type, be.amount
    FROM billing_events be
    JOIN cohort co ON co.customer_id = be.customer_id
    WHERE be.event_date < :reference_date
),

billing_rfm AS (
    SELECT
        customer_id,
        MAX(event_date) FILTER (WHERE event_type = 'charge') AS last_charge_date,
        COUNT(*) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '30 days'
        ) AS frequency_charges_30d,
        COUNT(*) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '60 days'
        ) AS frequency_charges_60d,
        COUNT(*) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '90 days'
        ) AS frequency_charges_90d,
        COALESCE(SUM(amount) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '30 days'
        ), 0) AS monetary_charges_30d,
        COALESCE(SUM(amount) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '60 days'
        ), 0) AS monetary_charges_60d,
        COALESCE(SUM(amount) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '90 days'
        ), 0) AS monetary_charges_90d,
        -- Replaces the leaky raw TotalCharges column — see
        -- DECISIONS.md#total-charges-dropped-at-ingestion.
        COALESCE(SUM(amount) FILTER (WHERE event_type = 'charge'), 0) AS cumulative_charges_to_date,
        COUNT(*) FILTER (
            WHERE event_type = 'payment_failed' AND event_date >= :reference_date - INTERVAL '180 days'
        ) AS failed_180d,
        COUNT(*) FILTER (
            WHERE event_type = 'charge' AND event_date >= :reference_date - INTERVAL '180 days'
        ) AS cycles_180d
    FROM billing_history
    GROUP BY customer_id
),

-- Trend: OLS slope of charge amount over the trailing 90 days. The
-- dataset has no raw usage/minutes column, so this is a spend-trend
-- proxy, not a true usage trend — documented limitation (PLAN.md Phase 0
-- feature catalog). NULL when fewer than 2 points exist in the window
-- (regr_slope needs at least 2; HAVING drops the group entirely, and the
-- outer LEFT JOIN turns that into a NULL rather than an error).
billing_slope AS (
    SELECT
        customer_id,
        regr_slope(amount, EXTRACT(EPOCH FROM event_date) / 86400.0) AS billing_amount_slope_90d
    FROM billing_history
    WHERE event_type = 'charge'
      AND event_date >= :reference_date - INTERVAL '90 days'
    GROUP BY customer_id
    HAVING COUNT(*) >= 2
),

-- Month-over-month change in total charges, via LAG over each customer's
-- monthly billing time series (only months fully before reference_date).
monthly_charge_totals AS (
    SELECT
        customer_id,
        date_trunc('month', event_date) AS charge_month,
        SUM(amount) AS month_total
    FROM billing_history
    WHERE event_type = 'charge'
    GROUP BY customer_id, date_trunc('month', event_date)
),
monthly_charge_trend AS (
    SELECT
        customer_id,
        month_total,
        LAG(month_total) OVER (PARTITION BY customer_id ORDER BY charge_month) AS prior_month_total,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY charge_month DESC) AS months_ago
    FROM monthly_charge_totals
),
billing_mom_delta AS (
    SELECT customer_id, month_total - prior_month_total AS billing_amount_mom_delta
    FROM monthly_charge_trend
    WHERE months_ago = 1 AND prior_month_total IS NOT NULL
),

-- Payment-failure streak: how many of the most recent consecutive
-- billing cycles (ordered by event_date DESC) had a payment_failed
-- event, stopping at the first cycle that didn't — classic
-- gaps-and-islands via ROW_NUMBER + a FILTERed window MIN.
billing_cycle_status AS (
    SELECT
        customer_id,
        event_date,
        BOOL_OR(event_type = 'payment_failed') AS cycle_failed,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY event_date DESC) AS cycles_ago
    FROM billing_history
    WHERE event_type IN ('charge', 'payment_failed')
    GROUP BY customer_id, event_date
),
payment_failure_streak AS (
    SELECT customer_id, COUNT(*) FILTER (WHERE cycle_failed) AS payment_failure_streak
    FROM (
        SELECT
            customer_id, cycle_failed, cycles_ago,
            MIN(cycles_ago) FILTER (WHERE NOT cycle_failed)
                OVER (PARTITION BY customer_id) AS first_ok_cycle
        FROM billing_cycle_status
    ) ranked
    WHERE cycles_ago < COALESCE(first_ok_cycle, cycles_ago + 1)
    GROUP BY customer_id
),

-- ---------------------------------------------------------------------
-- Support-ticket-derived features, same point-in-time filter.
-- ---------------------------------------------------------------------
ticket_history AS (
    SELECT st.customer_id, st.created_date
    FROM support_tickets st
    JOIN cohort co ON co.customer_id = st.customer_id
    WHERE st.created_date < :reference_date
),
ticket_rfm AS (
    SELECT
        customer_id,
        MAX(created_date) AS last_ticket_date,
        COUNT(*) FILTER (WHERE created_date >= :reference_date - INTERVAL '30 days') AS frequency_tickets_30d,
        COUNT(*) FILTER (WHERE created_date >= :reference_date - INTERVAL '60 days') AS frequency_tickets_60d,
        COUNT(*) FILTER (WHERE created_date >= :reference_date - INTERVAL '90 days') AS frequency_tickets_90d,
        COUNT(*) FILTER (
            WHERE created_date >= :reference_date - INTERVAL '60 days'
              AND created_date < :reference_date - INTERVAL '30 days'
        ) AS tickets_prior_30_60d
    FROM ticket_history
    GROUP BY customer_id
)

SELECT
    co.customer_id,
    CAST(:reference_date AS date) AS reference_date,
    :feature_set_version AS feature_set_version,

    co.senior_citizen,
    co.partner,
    co.dependents,
    co.contract_type,
    co.internet_service,
    co.online_security,
    co.online_backup,
    co.device_protection,
    co.tech_support,
    co.streaming_tv,
    co.streaming_movies,
    co.paperless_billing,
    co.payment_method,
    co.monthly_charges,

    -- Tenure recomputed as of reference_date — never the raw `tenure`
    -- column (DECISIONS.md#leakage-mitigations).
    (
        DATE_PART('year', AGE(CAST(:reference_date AS date), co.signup_date)) * 12
        + DATE_PART('month', AGE(CAST(:reference_date AS date), co.signup_date))
    )::int AS tenure_months,
    CASE
        WHEN (
            DATE_PART('year', AGE(CAST(:reference_date AS date), co.signup_date)) * 12
            + DATE_PART('month', AGE(CAST(:reference_date AS date), co.signup_date))
        ) < 6 THEN '0-6'
        WHEN (
            DATE_PART('year', AGE(CAST(:reference_date AS date), co.signup_date)) * 12
            + DATE_PART('month', AGE(CAST(:reference_date AS date), co.signup_date))
        ) < 12 THEN '6-12'
        WHEN (
            DATE_PART('year', AGE(CAST(:reference_date AS date), co.signup_date)) * 12
            + DATE_PART('month', AGE(CAST(:reference_date AS date), co.signup_date))
        ) < 24 THEN '12-24'
        WHEN (
            DATE_PART('year', AGE(CAST(:reference_date AS date), co.signup_date)) * 12
            + DATE_PART('month', AGE(CAST(:reference_date AS date), co.signup_date))
        ) < 48 THEN '24-48'
        ELSE '48+'
    END AS tenure_bucket,

    (CAST(:reference_date AS date) - br.last_charge_date) AS recency_days_since_last_charge,
    (CAST(:reference_date AS date) - tr.last_ticket_date) AS recency_days_since_last_ticket,
    COALESCE(br.frequency_charges_30d, 0) AS frequency_charges_30d,
    COALESCE(br.frequency_charges_60d, 0) AS frequency_charges_60d,
    COALESCE(br.frequency_charges_90d, 0) AS frequency_charges_90d,
    COALESCE(tr.frequency_tickets_30d, 0) AS frequency_tickets_30d,
    COALESCE(tr.frequency_tickets_60d, 0) AS frequency_tickets_60d,
    COALESCE(tr.frequency_tickets_90d, 0) AS frequency_tickets_90d,
    COALESCE(br.monetary_charges_30d, 0) AS monetary_charges_30d,
    COALESCE(br.monetary_charges_60d, 0) AS monetary_charges_60d,
    COALESCE(br.monetary_charges_90d, 0) AS monetary_charges_90d,
    COALESCE(br.cumulative_charges_to_date, 0) AS cumulative_charges_to_date,

    bs.billing_amount_slope_90d,
    bmd.billing_amount_mom_delta,
    (COALESCE(tr.frequency_tickets_30d, 0) - COALESCE(tr.tickets_prior_30_60d, 0)) AS ticket_velocity_delta_30_60,
    COALESCE(pfs.payment_failure_streak, 0) AS payment_failure_streak,
    CASE
        WHEN COALESCE(br.cycles_180d, 0) = 0 THEN NULL
        ELSE br.failed_180d::numeric / br.cycles_180d
    END AS payment_failure_rate_180d,

    -- Label — see DECISIONS.md#label-lookahead-is-not-leakage. Looks
    -- forward from reference_date on purpose; never joined into any
    -- feature CTE above.
    (
        co.churn_date IS NOT NULL
        AND co.churn_date > CAST(:reference_date AS date)
        AND co.churn_date <= CAST(:reference_date AS date) + make_interval(days => :horizon_days)
    ) AS label_churn_next_30d

FROM cohort co
LEFT JOIN billing_rfm br ON br.customer_id = co.customer_id
LEFT JOIN billing_slope bs ON bs.customer_id = co.customer_id
LEFT JOIN billing_mom_delta bmd ON bmd.customer_id = co.customer_id
LEFT JOIN payment_failure_streak pfs ON pfs.customer_id = co.customer_id
LEFT JOIN ticket_rfm tr ON tr.customer_id = co.customer_id;
