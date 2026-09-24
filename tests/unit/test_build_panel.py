import datetime as dt
from unittest.mock import MagicMock

from churn.features.build_panel import _already_built, _delete_existing, _monthly_reference_dates


def test_monthly_reference_dates_includes_both_endpoints() -> None:
    dates = _monthly_reference_dates(dt.date(2023, 1, 1), dt.date(2023, 4, 1))
    assert dates == [
        dt.date(2023, 1, 1),
        dt.date(2023, 2, 1),
        dt.date(2023, 3, 1),
        dt.date(2023, 4, 1),
    ]


def test_monthly_reference_dates_single_month_when_start_equals_end() -> None:
    assert _monthly_reference_dates(dt.date(2024, 6, 1), dt.date(2024, 6, 1)) == [
        dt.date(2024, 6, 1)
    ]


def test_monthly_reference_dates_steps_by_calendar_month_not_30_days() -> None:
    # Jan (31 days) -> Feb -> Mar (31 days): a fixed-day-count step would
    # drift off the 1st; relativedelta must not.
    dates = _monthly_reference_dates(dt.date(2024, 1, 1), dt.date(2024, 3, 1))
    assert all(d.day == 1 for d in dates)
    assert dates[1] == dt.date(2024, 2, 1)


def test_already_built_true_when_a_row_exists() -> None:
    conn = MagicMock()
    conn.execute.return_value.first.return_value = (1,)
    assert _already_built(conn, dt.date(2023, 1, 1), "v1") is True


def test_already_built_false_when_no_row_exists() -> None:
    conn = MagicMock()
    conn.execute.return_value.first.return_value = None
    assert _already_built(conn, dt.date(2023, 1, 1), "v1") is False


def test_delete_existing_executes_a_delete_scoped_to_reference_date_and_version() -> None:
    conn = MagicMock()
    _delete_existing(conn, dt.date(2023, 1, 1), "v1")

    conn.execute.assert_called_once()
    executed_params = conn.execute.call_args.args[1]
    assert executed_params == {"rd": dt.date(2023, 1, 1), "v": "v1"}
