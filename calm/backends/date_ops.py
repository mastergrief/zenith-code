"""
CALM date/time backend — verified temporal computations.

The model writes "there are 45 days between March 1 and April 15"
and Auto-CALM verifies it on CPU.

Functions:
  days_between(date1, date2)   — days between two dates
  day_of_week(date)            — Monday/Tuesday/etc
  is_leap_year(year)           — True/False
  days_in_month(month, year)   — 28/29/30/31
  add_days(date, n)            — date + n days
  date_diff(date1, date2)      — {years, months, days}
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple, Union


def _parse_date(s: str) -> date:
    """Parse a date string. Supports YYYY-MM-DD and common formats."""
    s = s.strip().strip('"').strip("'")
    # Try ISO format first.
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse date: {s}")


def days_between(d1: str, d2: str) -> int:
    """Number of days between two dates (absolute value)."""
    return abs((_parse_date(d2) - _parse_date(d1)).days)


def day_of_week(d: str) -> str:
    """Day of the week for a date."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[_parse_date(d).weekday()]


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    year = int(year)
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_month(month: int, year: int = 2024) -> int:
    """Number of days in a given month."""
    month, year = int(month), int(year)
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    raise ValueError(f"invalid month: {month}")


def add_days(d: str, n: int) -> str:
    """Add n days to a date, return ISO string."""
    result = _parse_date(d) + timedelta(days=int(n))
    return result.isoformat()


def date_diff(d1: str, d2: str) -> dict:
    """Difference between two dates as {years, months, days}."""
    a, b = _parse_date(d1), _parse_date(d2)
    if a > b:
        a, b = b, a
    years = b.year - a.year
    months = b.month - a.month
    days = b.day - a.day
    if days < 0:
        months -= 1
        # Days in the previous month of b.
        prev_month = b.month - 1 if b.month > 1 else 12
        prev_year = b.year if b.month > 1 else b.year - 1
        days += days_in_month(prev_month, prev_year)
    if months < 0:
        years -= 1
        months += 12
    return {"years": years, "months": months, "days": days}


# Registry for expression.py integration.
DATE_FUNCTIONS = {
    "days_between": days_between,
    "day_of_week": day_of_week,
    "is_leap_year": is_leap_year,
    "days_in_month": days_in_month,
    "add_days": add_days,
    "date_diff": date_diff,
}
