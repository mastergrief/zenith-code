"""
CALM cron backend — verified cron expression parsing and scheduling.

Nobody can read "0 */4 * * 1-5" reliably. This backend parses cron
expressions, explains them in English, and computes next run times.

Functions: cron_parse, cron_explain, cron_next, cron_matches, cron_validate, cron_frequency.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional


_FIELD_NAMES = ["minute", "hour", "day_of_month", "month", "day_of_week"]
_FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 0=Sunday in standard cron.
}

_MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_DOW_NAMES = {
    "SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6,
}


def _replace_names(expr: str) -> str:
    """Replace month/dow names with numbers."""
    upper = expr.upper()
    for name, num in _MONTH_NAMES.items():
        upper = upper.replace(name, str(num))
    for name, num in _DOW_NAMES.items():
        upper = upper.replace(name, str(num))
    return upper


def _expand_field(field: str, lo: int, hi: int) -> List[int]:
    """Expand a single cron field into a sorted list of values."""
    if field == "*":
        return list(range(lo, hi + 1))

    values = set()
    for part in field.split(","):
        part = part.strip()

        # Step: */N or M-N/S or M/S.
        step_match = re.match(r'^(\S+)/(\d+)$', part)
        step = None
        if step_match:
            part = step_match.group(1)
            step = int(step_match.group(2))
            if step == 0:
                step = 1

        # Range: M-N.
        range_match = re.match(r'^(\d+)-(\d+)$', part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            start = max(lo, min(hi, start))
            end = max(lo, min(hi, end))
            if step:
                values.update(range(start, end + 1, step))
            else:
                values.update(range(start, end + 1))
        elif part == "*":
            if step:
                values.update(range(lo, hi + 1, step))
            else:
                values.update(range(lo, hi + 1))
        else:
            try:
                v = int(part)
                if lo <= v <= hi:
                    values.add(v)
            except ValueError:
                pass

    return sorted(values)


def cron_parse(expression: str) -> dict:
    """Parse a cron expression into its component fields.
    Example: cron_parse("30 */4 * * 1-5")
    → {minute: [30], hour: [0,4,8,12,16,20], day_of_month: [1..31], month: [1..12], day_of_week: [1,2,3,4,5]}"""
    expr = _replace_names(expression.strip())
    parts = expr.split()

    # Handle @shortcuts.
    shortcuts = {
        "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
    }
    if len(parts) == 1 and parts[0].lower() in shortcuts:
        parts = shortcuts[parts[0].lower()].split()

    if len(parts) != 5:
        return {"valid": False, "error": f"Expected 5 fields, got {len(parts)}"}

    result = {"valid": True, "raw": expression}
    for i, name in enumerate(_FIELD_NAMES):
        lo, hi = _FIELD_RANGES[name]
        result[name] = _expand_field(parts[i], lo, hi)

    return result


def cron_explain(expression: str) -> str:
    """Explain a cron expression in plain English.
    Example: cron_explain("0 9 * * 1-5") → "At 09:00, Monday through Friday" """
    parsed = cron_parse(expression)
    if not parsed.get("valid"):
        return parsed.get("error", "Invalid cron expression")

    parts = []

    # Time.
    mins = parsed["minute"]
    hours = parsed["hour"]
    if len(mins) == 1 and len(hours) == 1:
        parts.append(f"At {hours[0]:02d}:{mins[0]:02d}")
    elif len(mins) == 1:
        parts.append(f"At minute {mins[0]}")
        if len(hours) < 24:
            parts.append(_describe_list(hours, "hour"))
    elif len(hours) == 1:
        parts.append(f"During hour {hours[0]:02d}")
        if len(mins) < 60:
            parts.append(_describe_list(mins, "minute"))
    else:
        if len(mins) < 60:
            parts.append(_describe_list(mins, "minute"))
        if len(hours) < 24:
            parts.append(_describe_list(hours, "hour"))

    # Day of week.
    dow = parsed["day_of_week"]
    dow_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    if len(dow) < 7:
        if dow == list(range(1, 6)):
            parts.append("Monday through Friday")
        elif dow == [0, 6]:
            parts.append("weekends")
        else:
            parts.append(", ".join(dow_names[d] for d in dow))

    # Day of month.
    dom = parsed["day_of_month"]
    if len(dom) < 31:
        parts.append(f"on day{'s' if len(dom) > 1 else ''} {_compact_range(dom)} of the month")

    # Month.
    months = parsed["month"]
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    if len(months) < 12:
        parts.append("in " + ", ".join(month_names[m] for m in months))

    return ", ".join(parts) if parts else "Every minute"


def cron_next(expression: str, count: int = 5, after: str = "") -> list:
    """Compute the next N run times for a cron expression.
    Example: cron_next("0 9 * * 1-5", 3) → ["2026-04-13 09:00", "2026-04-14 09:00", ...]"""
    parsed = cron_parse(expression)
    if not parsed.get("valid"):
        return [parsed.get("error", "Invalid")]

    count = min(int(count), 20)  # Cap at 20.

    if after:
        try:
            dt = datetime.strptime(after, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                dt = datetime.strptime(after, "%Y-%m-%dT%H:%M")
            except ValueError:
                dt = datetime.now()
    else:
        dt = datetime.now()

    dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
    results = []

    for _ in range(525960):  # Max ~1 year of minutes.
        if (dt.minute in parsed["minute"] and
                dt.hour in parsed["hour"] and
                dt.day in parsed["day_of_month"] and
                dt.month in parsed["month"] and
                dt.weekday() in _isoweekday_to_cron(parsed["day_of_week"])):
            results.append(dt.strftime("%Y-%m-%d %H:%M"))
            if len(results) >= count:
                break
        dt += timedelta(minutes=1)

    return results


def cron_matches(expression: str, timestamp: str) -> bool:
    """Check if a timestamp matches a cron expression.
    Example: cron_matches("0 9 * * 1", "2026-04-13 09:00") → True (Monday at 9am)"""
    parsed = cron_parse(expression)
    if not parsed.get("valid"):
        return False

    try:
        dt = datetime.strptime(timestamp.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            dt = datetime.strptime(timestamp.strip(), "%Y-%m-%dT%H:%M")
        except ValueError:
            return False

    return (dt.minute in parsed["minute"] and
            dt.hour in parsed["hour"] and
            dt.day in parsed["day_of_month"] and
            dt.month in parsed["month"] and
            dt.weekday() in _isoweekday_to_cron(parsed["day_of_week"]))


def cron_validate(expression: str) -> dict:
    """Validate a cron expression and report errors.
    Returns {valid, errors, field_count}."""
    expr = _replace_names(expression.strip())
    parts = expr.split()

    # Check @shortcuts.
    if len(parts) == 1 and parts[0].startswith("@"):
        shortcuts = {"@yearly", "@annually", "@monthly", "@weekly", "@daily", "@midnight", "@hourly"}
        if parts[0].lower() in shortcuts:
            return {"valid": True, "errors": [], "field_count": 5}
        return {"valid": False, "errors": [f"Unknown shortcut: {parts[0]}"], "field_count": 0}

    if len(parts) != 5:
        return {"valid": False, "errors": [f"Expected 5 fields, got {len(parts)}"], "field_count": len(parts)}

    errors = []
    for i, name in enumerate(_FIELD_NAMES):
        lo, hi = _FIELD_RANGES[name]
        expanded = _expand_field(parts[i], lo, hi)
        if not expanded:
            errors.append(f"{name}: '{parts[i]}' produces no valid values (range {lo}-{hi})")

    return {"valid": len(errors) == 0, "errors": errors, "field_count": 5}


def cron_frequency(expression: str) -> dict:
    """Estimate how often a cron job runs.
    Returns {runs_per_day, runs_per_week, runs_per_month, description}."""
    parsed = cron_parse(expression)
    if not parsed.get("valid"):
        return {"error": parsed.get("error", "Invalid")}

    mins = len(parsed["minute"])
    hours = len(parsed["hour"])
    doms = len(parsed["day_of_month"])
    months = len(parsed["month"])
    dows = len(parsed["day_of_week"])

    # Runs per day (approximate — ignoring dom/dow interaction).
    per_day = mins * hours
    per_week = per_day * min(dows, 7)
    per_month = per_day * min(doms, 30) * (months / 12)

    if per_day >= 1440:
        desc = "every minute"
    elif per_day >= 60:
        desc = f"~{per_day} times/day"
    elif per_day > 1:
        desc = f"{per_day} times/day"
    elif per_day == 1:
        desc = "once daily"
    else:
        desc = "less than daily"

    return {
        "runs_per_day": round(per_day, 1),
        "runs_per_week": round(per_week, 1),
        "runs_per_month": round(per_month, 1),
        "description": desc,
    }


def _isoweekday_to_cron(cron_days: List[int]) -> set:
    """Convert cron day-of-week (0=Sun) to Python weekday (0=Mon)."""
    # Cron: 0=Sun,1=Mon,...,6=Sat → Python: 0=Mon,...,6=Sun.
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return {mapping[d] for d in cron_days if d in mapping}


def _describe_list(values: list, unit: str) -> str:
    """Describe a list of values compactly."""
    if len(values) <= 3:
        return f"at {unit}{'s' if len(values) > 1 else ''} " + ", ".join(str(v) for v in values)
    return f"at {len(values)} {unit}s"


def _compact_range(values: list) -> str:
    """Compact a sorted list into range notation: [1,2,3,5,7,8] → '1-3, 5, 7-8'."""
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = v
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


CRON_FUNCTIONS = {
    "cron_parse": cron_parse,
    "cron_explain": cron_explain,
    "cron_next": cron_next,
    "cron_matches": cron_matches,
    "cron_validate": cron_validate,
    "cron_frequency": cron_frequency,
}
