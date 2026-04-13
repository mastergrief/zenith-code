"""
CALM Time/date backend — epoch conversion, business days, date math.

Models botch epoch math, mess up business day calculations, confuse time zones.
Extends date_ops with more advanced operations.
"""

from __future__ import annotations

import datetime
import calendar


def epoch_to_date(epoch: float) -> str:
    """Convert Unix epoch timestamp to ISO 8601 UTC datetime string."""
    dt = datetime.datetime.fromtimestamp(float(epoch), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def date_to_epoch(date_str: str) -> int:
    """Convert ISO 8601 date string to Unix epoch. Supports YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS."""
    s = str(date_str).strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return -1


def business_days(start: str, end: str) -> int:
    """Count business days (Mon-Fri) between two dates (YYYY-MM-DD), exclusive of end."""
    d1 = datetime.date.fromisoformat(str(start))
    d2 = datetime.date.fromisoformat(str(end))
    if d1 > d2:
        d1, d2 = d2, d1
    count = 0
    current = d1
    while current < d2:
        if current.weekday() < 5:
            count += 1
        current += datetime.timedelta(days=1)
    return count


def add_business_days(start: str, days: int) -> str:
    """Add N business days to a date. Returns YYYY-MM-DD."""
    current = datetime.date.fromisoformat(str(start))
    remaining = int(days)
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current.isoformat()


def days_in_month(year: int, month: int) -> int:
    """Number of days in a given month/year."""
    return calendar.monthrange(int(year), int(month))[1]


def is_weekend(date_str: str) -> bool:
    """Whether a date falls on Saturday or Sunday."""
    d = datetime.date.fromisoformat(str(date_str))
    return d.weekday() >= 5


def quarter(date_str: str) -> int:
    """Fiscal quarter (1-4) for a date."""
    d = datetime.date.fromisoformat(str(date_str))
    return (d.month - 1) // 3 + 1


def week_number(date_str: str) -> int:
    """ISO week number for a date."""
    d = datetime.date.fromisoformat(str(date_str))
    return d.isocalendar()[1]


def age(birthdate: str, as_of: str = None) -> dict:
    """Calculate age in years, months, days from birthdate."""
    birth = datetime.date.fromisoformat(str(birthdate))
    today = datetime.date.fromisoformat(str(as_of)) if as_of else datetime.date.today()
    years = today.year - birth.year
    months = today.month - birth.month
    days = today.day - birth.day
    if days < 0:
        months -= 1
        days += calendar.monthrange(today.year, today.month - 1 if today.month > 1 else 12)[1]
    if months < 0:
        years -= 1
        months += 12
    return {"years": years, "months": months, "days": days}


def time_until(target: str) -> dict:
    """Time remaining until a target date from today."""
    t = datetime.date.fromisoformat(str(target))
    today = datetime.date.today()
    delta = t - today
    total = delta.days
    if total < 0:
        return {"days": total, "weeks": round(total / 7, 1), "status": "past"}
    return {"days": total, "weeks": round(total / 7, 1), "months": round(total / 30.44, 1), "status": "future"}


def overlap_days(start1: str, end1: str, start2: str, end2: str) -> int:
    """Number of overlapping days between two date ranges."""
    s1, e1 = datetime.date.fromisoformat(str(start1)), datetime.date.fromisoformat(str(end1))
    s2, e2 = datetime.date.fromisoformat(str(start2)), datetime.date.fromisoformat(str(end2))
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    return max(0, (overlap_end - overlap_start).days + 1)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    s = float(seconds)
    if s < 0:
        return f"-{format_duration(-s)}"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m = int(s // 60)
        sec = s % 60
        return f"{m}m {sec:.0f}s"
    if s < 86400:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h {m}m"
    d = int(s // 86400)
    h = int((s % 86400) // 3600)
    return f"{d}d {h}h"


TIME_FUNCTIONS = {
    "epoch_to_date": epoch_to_date,
    "date_to_epoch": date_to_epoch,
    "business_days": business_days,
    "add_business_days": add_business_days,
    "days_in_month": days_in_month,
    "is_weekend": is_weekend,
    "quarter": quarter,
    "week_number": week_number,
    "age": age,
    "time_until": time_until,
    "overlap_days": overlap_days,
    "format_duration": format_duration,
}

TIME_NL_PATTERNS = [
    (r'(?:convert|what is)\s+(?:epoch|timestamp)\s+([\d.]+)\s+(?:to|in|as)\s+(?:date|datetime|human)', 'epoch_to_date({0})'),
    (r'(?:epoch|timestamp)\s+(?:of|for)\s+(\d{4}-\d{2}-\d{2})', 'date_to_epoch("{0}")'),
    (r'(?:business|working|work)\s+days?\s+(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})', 'business_days("{0}", "{1}")'),
    (r'(?:is)\s+(\d{4}-\d{2}-\d{2})\s+(?:a\s+)?weekend', 'is_weekend("{0}")'),
    (r'(?:what|which)\s+quarter\s+(?:is)\s+(\d{4}-\d{2}-\d{2})', 'quarter("{0}")'),
    (r'(?:how many|number of)\s+days?\s+in\s+(\w+)\s+(\d{4})', None),
    (r'(?:how old|age)\s+(?:is someone|if)\s+born\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})', 'age("{0}")'),
    (r'(?:days?|time)\s+(?:until|remaining|left)\s+(\d{4}-\d{2}-\d{2})', 'time_until("{0}")'),
]
