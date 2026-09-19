-- Adversarial point-in-time check: independently re-derive
-- cumulative_charges_to_date and frequency_charges_30d directly from
-- billing_events for every stored feature_snapshots row, and flag any
-- mismatch against what build_feature_snapshot.sql actually wrote. A
-- mismatch here means either the extraction query used data on/after
-- reference_date, or the stored row has drifted from what it should be
-- (e.g. billing_events were edited after the snapshot was built) —
-- either way, a real bug, not a formatting nitpick. Should return zero
-- rows.
--
-- Params: :feature_set_version

WITH recomputed AS (
    SELECT
        fs.snapshot_id,
        fs.customer_id,
        fs.reference_date,
        fs.cumulative_charges_to_date AS stored_cumulative,
        fs.frequency_charges_30d AS stored_freq_30d,
        COALESCE(SUM(be.amount) FILTER (
            WHERE be.event_type = 'charge' AND be.event_date < fs.reference_date
        ), 0) AS recomputed_cumulative,
        COUNT(*) FILTER (
            WHERE be.event_type = 'charge'
              AND be.event_date < fs.reference_date
              AND be.event_date >= fs.reference_date - INTERVAL '30 days'
        ) AS recomputed_freq_30d
    FROM feature_snapshots fs
    LEFT JOIN billing_events be ON be.customer_id = fs.customer_id
    WHERE fs.feature_set_version = :feature_set_version
    GROUP BY fs.snapshot_id, fs.customer_id, fs.reference_date,
             fs.cumulative_charges_to_date, fs.frequency_charges_30d
)
SELECT *
FROM recomputed
WHERE stored_cumulative <> recomputed_cumulative
   OR stored_freq_30d <> recomputed_freq_30d;
