"""
CALM Calendar backend — day names, month info, Easter, zodiac, workday calculations.

Models botch day-of-week for historical dates, hallucinate month lengths.
"""

from __future__ import annotations

import datetime
import calendar


_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_ZODIAC = [
    ((1, 20), (2, 18), "Aquarius"), ((2, 19), (3, 20), "Pisces"),
    ((3, 21), (4, 19), "Aries"), ((4, 20), (5, 20), "Taurus"),
    ((5, 21), (6, 20), "Gemini"), ((6, 21), (7, 22), "Cancer"),
    ((7, 23), (8, 22), "Leo"), ((8, 23), (9, 22), "Virgo"),
    ((9, 23), (10, 22), "Libra"), ((10, 23), (11, 21), "Scorpio"),
    ((11, 22), (12, 21), "Sagittarius"), ((12, 22), (1, 19), "Capricorn"),
]


def day_of_week(date_str: str) -> str:
    """Day of the week for a date (YYYY-MM-DD). E.g. '2026-04-13' → 'Monday'."""
    d = datetime.date.fromisoformat(str(date_str))
    return _DAY_NAMES[d.weekday()]


def month_name(month: int) -> str:
    """Name of a month (1-12)."""
    m = int(month)
    if 1 <= m <= 12:
        return _MONTH_NAMES[m - 1]
    return f"Invalid month: {month}"


def days_in_year(year: int) -> int:
    """Number of days in a year (365 or 366)."""
    return 366 if calendar.isleap(int(year)) else 365


def is_leap_year(year: int) -> bool:
    """Whether a year is a leap year."""
    return calendar.isleap(int(year))


def easter(year: int) -> str:
    """Date of Easter Sunday for a given year (Computus algorithm). Returns YYYY-MM-DD."""
    y = int(year)
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return f"{y}-{month:02d}-{day:02d}"


def zodiac_sign(month: int, day: int) -> str:
    """Western zodiac sign for a date."""
    m, d = int(month), int(day)
    for (sm, sd), (em, ed), sign in _ZODIAC:
        if sm == em:
            if m == sm and sd <= d <= ed:
                return sign
        elif sm < em:
            if (m == sm and d >= sd) or (m == em and d <= ed):
                return sign
        else:  # wraps around year (Capricorn)
            if (m == sm and d >= sd) or (m == em and d <= ed):
                return sign
    return "Capricorn"


def nth_weekday(year: int, month: int, weekday: int, n: int) -> str:
    """Nth occurrence of a weekday in a month. weekday: 0=Mon...6=Sun. Returns YYYY-MM-DD."""
    y, m, wd, count = int(year), int(month), int(weekday), int(n)
    first = datetime.date(y, m, 1)
    diff = (wd - first.weekday()) % 7
    target = first + datetime.timedelta(days=diff + 7 * (count - 1))
    if target.month != m:
        return "does not exist"
    return target.isoformat()


def weeks_between(date1: str, date2: str) -> float:
    """Number of weeks between two dates."""
    d1 = datetime.date.fromisoformat(str(date1))
    d2 = datetime.date.fromisoformat(str(date2))
    return round(abs((d2 - d1).days) / 7, 2)


def age_in_days(birthdate: str, as_of: str = None) -> int:
    """Age in days from birthdate."""
    birth = datetime.date.fromisoformat(str(birthdate))
    today = datetime.date.fromisoformat(str(as_of)) if as_of else datetime.date.today()
    return (today - birth).days


def day_of_year(date_str: str) -> int:
    """Day number within the year (1-366)."""
    d = datetime.date.fromisoformat(str(date_str))
    return d.timetuple().tm_yday


def days_remaining_in_year(date_str: str) -> int:
    """Days remaining in the year from a given date."""
    d = datetime.date.fromisoformat(str(date_str))
    total = days_in_year(d.year)
    return total - day_of_year(date_str)


def next_weekday(date_str: str, target_weekday: int) -> str:
    """Next occurrence of a weekday after a date. weekday: 0=Mon...6=Sun."""
    d = datetime.date.fromisoformat(str(date_str))
    tw = int(target_weekday)
    diff = (tw - d.weekday()) % 7
    if diff == 0:
        diff = 7  # next week
    return (d + datetime.timedelta(days=diff)).isoformat()


def month_calendar(year: int, month: int) -> str:
    """Text calendar for a month."""
    return calendar.month(int(year), int(month))


def is_workday(date_str: str) -> bool:
    """Whether a date is a workday (Mon-Fri). Does NOT account for holidays."""
    d = datetime.date.fromisoformat(str(date_str))
    return d.weekday() < 5


CALENDAR_FUNCTIONS = {
    "day_of_week": day_of_week,
    "month_name": month_name,
    "days_in_year": days_in_year,
    "is_leap_year": is_leap_year,
    "easter": easter,
    "zodiac_sign": zodiac_sign,
    "nth_weekday": nth_weekday,
    "weeks_between": weeks_between,
    "age_in_days": age_in_days,
    "day_of_year": day_of_year,
    "days_remaining_in_year": days_remaining_in_year,
    "next_weekday": next_weekday,
    "month_calendar": month_calendar,
    "is_workday": is_workday,
}

CALENDAR_NL_PATTERNS = [
    (r'(?:what|which)\s+day\s+(?:of the week\s+)?(?:is|was|will be)\s+(\d{4}-\d{2}-\d{2})', 'day_of_week("{0}")'),
    (r'(?:is)\s+(\d{4})\s+(?:a\s+)?leap\s+year', 'is_leap_year({0})'),
    (r'(?:when is|date of)\s+easter\s+(?:in\s+)?(\d{4})', 'easter({0})'),
    (r'(?:zodiac|star sign|horoscope)\s+(?:for|of)\s+(\w+)\s+(\d{1,2})', None),
    (r'(?:how many|number of)\s+weeks?\s+(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})', 'weeks_between("{0}", "{1}")'),
    (r'(?:what|which)\s+day\s+(?:of|in)\s+(?:the\s+)?year\s+(?:is)\s+(\d{4}-\d{2}-\d{2})', 'day_of_year("{0}")'),
    (r'(?:is)\s+(\d{4}-\d{2}-\d{2})\s+(?:a\s+)?(?:workday|business day|weekday)', 'is_workday("{0}")'),
]
