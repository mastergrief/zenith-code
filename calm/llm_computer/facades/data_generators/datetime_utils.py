"""DatetimeUtilsGenerator — date/time arithmetic with correctness gotchas.

Date/time code is where developers consistently make silent correctness
bugs: off-by-one days, forgotten timezones, DST skips, leap years,
month-length assumptions. Every entry here pins down a specific
correct pattern.

All solutions use stdlib (`datetime`, `calendar`, `time`) — no
timezone DB dependency (zoneinfo's data files aren't always present
in the sandbox). Aware-timezone tests use `datetime.timezone.utc`
and fixed-offset `timezone(timedelta(hours=h))`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


@dataclass
class DatetimeSpec:
    name: str
    signature: str
    problem: str
    solution: str
    test_cases: List[Tuple]
    algorithm: str
    complexity: str
    edge_cases: List[str]
    skip_sandbox: bool = False


def _specs() -> List[DatetimeSpec]:
    out: List[DatetimeSpec] = []

    out.append(DatetimeSpec(
        name="days_between",
        signature="def days_between(a, b):",
        problem="Write a Python function `days_between(a, b)` that returns the number of full days between two 'YYYY-MM-DD' strings, `b - a`. Returns an int (may be negative).",
        solution=(
            "def days_between(a, b):\n"
            "    from datetime import date\n"
            "    da = date.fromisoformat(a)\n"
            "    db = date.fromisoformat(b)\n"
            "    return (db - da).days\n"
        ),
        test_cases=[
            ("2024-01-01", "2024-01-02", 1),
            ("2024-01-02", "2024-01-01", -1),
            ("2024-01-01", "2024-01-01", 0),
            ("2024-02-28", "2024-03-01", 2),              # leap year
            ("2023-02-28", "2023-03-01", 1),              # non-leap year
            ("2024-01-01", "2025-01-01", 366),            # 2024 is leap → 366
            ("2023-01-01", "2024-01-01", 365),
        ],
        algorithm="date.fromisoformat + timedelta .days",
        complexity="O(1)",
        edge_cases=["leap year 366 days", "negative when b < a", "same day returns 0"],
    ))

    out.append(DatetimeSpec(
        name="is_leap_year",
        signature="def is_leap_year(y):",
        problem="Write a Python function `is_leap_year(y)` that returns True if y is a Gregorian leap year. Rule: divisible by 4, EXCEPT centuries not divisible by 400.",
        solution=(
            "def is_leap_year(y):\n"
            "    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"
        ),
        test_cases=[
            (2024, True),
            (2023, False),
            (2000, True),            # divisible by 400
            (1900, False),            # century NOT divisible by 400
            (2100, False),
            (2400, True),
            (4, True),
            (1, False),
        ],
        algorithm="y % 4 == 0 AND (y % 100 != 0 OR y % 400 == 0)",
        complexity="O(1)",
        edge_cases=["century rule (1900 not leap)", "400-year rule (2000 leap)", "year 0", "small years"],
    ))

    out.append(DatetimeSpec(
        name="day_of_week",
        signature="def day_of_week(ymd):",
        problem="Write a Python function `day_of_week(ymd)` that takes an ISO-date string and returns the English weekday name (Monday, Tuesday, ..., Sunday).",
        solution=(
            "def day_of_week(ymd):\n"
            "    from datetime import date\n"
            "    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',\n"
            "             'Friday', 'Saturday', 'Sunday']\n"
            "    return names[date.fromisoformat(ymd).weekday()]\n"
        ),
        test_cases=[
            ("2024-01-01", "Monday"),
            ("2024-01-02", "Tuesday"),
            ("2024-12-25", "Wednesday"),
            ("2024-07-04", "Thursday"),
            ("2000-01-01", "Saturday"),
            ("1970-01-01", "Thursday"),
        ],
        algorithm="date.weekday() index (0=Mon) into name list",
        complexity="O(1)",
        edge_cases=["weekday() uses 0=Mon (isoweekday uses 1=Mon)", "epoch is Thursday"],
    ))

    out.append(DatetimeSpec(
        name="days_in_month",
        signature="def days_in_month(y, m):",
        problem="Write a Python function `days_in_month(y, m)` using calendar.monthrange that returns the number of days in month m of year y. Handles leap-year February automatically.",
        solution=(
            "def days_in_month(y, m):\n"
            "    import calendar\n"
            "    return calendar.monthrange(y, m)[1]\n"
        ),
        test_cases=[
            (2024, 2, 29),     # leap Feb
            (2023, 2, 28),     # non-leap Feb
            (2024, 1, 31),
            (2024, 4, 30),
            (2024, 12, 31),
            (2000, 2, 29),     # 400-year leap
            (1900, 2, 28),     # century non-leap
        ],
        algorithm="calendar.monthrange (returns (first_weekday, n_days))",
        complexity="O(1)",
        edge_cases=["leap February", "30-day vs 31-day months", "December wrap (still 31)"],
    ))

    out.append(DatetimeSpec(
        name="add_months",
        signature="def add_months(ymd, n):",
        problem="Write a Python function `add_months(ymd, n)` that adds n months to an ISO-date. Clamp day-of-month if the target month is shorter (e.g. Jan 31 + 1 month = Feb 28/29, not Mar 3).",
        solution=(
            "def add_months(ymd, n):\n"
            "    from datetime import date\n"
            "    import calendar\n"
            "    d = date.fromisoformat(ymd)\n"
            "    m_total = d.year * 12 + (d.month - 1) + n\n"
            "    ny, nm = divmod(m_total, 12)\n"
            "    nm += 1\n"
            "    last = calendar.monthrange(ny, nm)[1]\n"
            "    return date(ny, nm, min(d.day, last)).isoformat()\n"
        ),
        test_cases=[
            ("2024-01-15", 1, "2024-02-15"),
            ("2024-01-31", 1, "2024-02-29"),           # clamp to leap Feb
            ("2023-01-31", 1, "2023-02-28"),           # clamp to non-leap Feb
            ("2024-01-31", 13, "2025-02-28"),
            ("2024-03-15", -1, "2024-02-15"),
            ("2024-01-15", 0, "2024-01-15"),
            ("2024-05-31", 1, "2024-06-30"),           # June has 30 days
        ],
        algorithm="total-months math + day clamp via monthrange",
        complexity="O(1)",
        edge_cases=["Jan 31 + 1m → Feb 28/29 (clamp)", "negative n goes backward", "zero n identity"],
    ))

    out.append(DatetimeSpec(
        name="start_of_week",
        signature="def start_of_week(ymd):",
        problem="Write a Python function `start_of_week(ymd)` that returns the Monday on or before the given ISO date, as an ISO string. Uses weekday()=0=Monday convention.",
        solution=(
            "def start_of_week(ymd):\n"
            "    from datetime import date, timedelta\n"
            "    d = date.fromisoformat(ymd)\n"
            "    return (d - timedelta(days=d.weekday())).isoformat()\n"
        ),
        test_cases=[
            ("2024-01-01", "2024-01-01"),        # already Monday
            ("2024-01-02", "2024-01-01"),        # Tuesday → Monday
            ("2024-01-07", "2024-01-01"),        # Sunday → previous Monday
            ("2024-01-08", "2024-01-08"),        # next Monday
            ("2024-03-03", "2024-02-26"),        # Sunday → Monday of prior week
        ],
        algorithm="subtract weekday() days via timedelta",
        complexity="O(1)",
        edge_cases=["Sunday resolves to previous Monday, not next", "already-Monday unchanged"],
    ))

    out.append(DatetimeSpec(
        name="utc_to_offset",
        signature="def utc_to_offset(iso_utc, offset_hours):",
        problem="Write a Python function `utc_to_offset(iso_utc, offset_hours)` that takes a UTC ISO-8601 string ending in 'Z' or '+00:00' and returns the same instant as ISO-8601 with the given fixed offset (e.g. +5.5 for India). Return the ISO string.",
        solution=(
            "def utc_to_offset(iso_utc, offset_hours):\n"
            "    from datetime import datetime, timedelta, timezone\n"
            "    s = iso_utc.replace('Z', '+00:00')\n"
            "    utc = datetime.fromisoformat(s)\n"
            "    tz = timezone(timedelta(hours=offset_hours))\n"
            "    return utc.astimezone(tz).isoformat()\n"
        ),
        test_cases=[
            ("2024-01-01T00:00:00Z", 0, "2024-01-01T00:00:00+00:00"),
            ("2024-01-01T00:00:00Z", 5, "2024-01-01T05:00:00+05:00"),
            ("2024-01-01T00:00:00+00:00", -8, "2023-12-31T16:00:00-08:00"),
            ("2024-06-15T12:00:00Z", 9, "2024-06-15T21:00:00+09:00"),
        ],
        algorithm="fromisoformat + astimezone(fixed-offset timezone)",
        complexity="O(1)",
        edge_cases=["'Z' suffix normalized to +00:00", "negative offsets cross day boundary", "fractional offsets possible via timedelta"],
    ))

    out.append(DatetimeSpec(
        name="iso_duration_parse",
        signature="def iso_duration_seconds(s):",
        problem="Write a Python function `iso_duration_seconds(s)` that parses a subset of ISO 8601 duration strings of the form 'PTnHnMnS' and returns total seconds as int. Accepts any non-negative combination of H, M, S; at least one required.",
        solution=(
            "def iso_duration_seconds(s):\n"
            "    import re\n"
            "    m = re.fullmatch(r'PT(?:(\\d+)H)?(?:(\\d+)M)?(?:(\\d+)S)?', s)\n"
            "    if m is None:\n"
            "        raise ValueError('invalid duration')\n"
            "    h, mi, se = m.groups()\n"
            "    if h is None and mi is None and se is None:\n"
            "        raise ValueError('at least one unit required')\n"
            "    return int(h or 0) * 3600 + int(mi or 0) * 60 + int(se or 0)\n"
        ),
        test_cases=[
            ("PT1H", 3600),
            ("PT30M", 1800),
            ("PT45S", 45),
            ("PT1H30M", 5400),
            ("PT2H15M30S", 8130),
            ("PT0H0M0S", 0),
        ],
        algorithm="anchored regex with optional H/M/S groups",
        complexity="O(|s|)",
        edge_cases=["only one unit allowed (PT1H)", "zero explicit units", "missing PT prefix rejected", "non-ASCII digits rejected"],
    ))

    out.append(DatetimeSpec(
        name="quarter_of_date",
        signature="def quarter_of(ymd):",
        problem="Write a Python function `quarter_of(ymd)` that returns the fiscal quarter (1-4) of the given ISO date. Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec.",
        solution=(
            "def quarter_of(ymd):\n"
            "    from datetime import date\n"
            "    m = date.fromisoformat(ymd).month\n"
            "    return (m - 1) // 3 + 1\n"
        ),
        test_cases=[
            ("2024-01-01", 1),
            ("2024-03-31", 1),
            ("2024-04-01", 2),
            ("2024-06-30", 2),
            ("2024-07-01", 3),
            ("2024-09-30", 3),
            ("2024-10-01", 4),
            ("2024-12-31", 4),
        ],
        algorithm="integer math: (month - 1) // 3 + 1",
        complexity="O(1)",
        edge_cases=["quarter boundary days (Mar 31, Apr 1)", "Dec 31 is Q4"],
    ))

    out.append(DatetimeSpec(
        name="unix_to_iso",
        signature="def unix_to_iso(ts):",
        problem="Write a Python function `unix_to_iso(ts)` that converts a Unix timestamp (seconds since 1970-01-01 UTC) to an ISO-8601 UTC string like '2024-01-01T00:00:00+00:00'. Handles negative timestamps (pre-epoch).",
        solution=(
            "def unix_to_iso(ts):\n"
            "    from datetime import datetime, timezone\n"
            "    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()\n"
        ),
        test_cases=[
            (0, "1970-01-01T00:00:00+00:00"),
            (1704067200, "2024-01-01T00:00:00+00:00"),
            (-86400, "1969-12-31T00:00:00+00:00"),
            (1, "1970-01-01T00:00:01+00:00"),
        ],
        algorithm="datetime.fromtimestamp with tz=timezone.utc",
        complexity="O(1)",
        edge_cases=["epoch (0)", "negative timestamp (pre-1970)", "aware UTC output with +00:00 suffix"],
    ))

    return out


class DatetimeUtilsGenerator(DomainDataGenerator):
    """Date/time arithmetic patterns with leap-year, month-clamp, and
    timezone correctness baked in. 10 canonical operations every
    datetime-touching codebase needs."""

    name = "datetime_utils"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        self.rng.shuffle(self._specs)
        out: List[VerifiedExample] = []
        for s in self._specs[:n]:
            out.append(VerifiedExample(
                problem=s.problem,
                signature=s.signature,
                solution=s.solution,
                test_cases=list(s.test_cases),
                reasoning="",
                algorithm=s.algorithm,
                complexity=s.complexity,
                edge_cases=list(s.edge_cases),
                category=f"dt_{s.name}",
                generator_name=self.name,
                skip_sandbox=s.skip_sandbox,
            ))
        return out


register_generator("datetime_utils", DatetimeUtilsGenerator)
