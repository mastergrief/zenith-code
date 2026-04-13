"""
CALM Timezone backend — convert, offset lookup, DST awareness.

Models are terrible at timezone math. Pure stdlib zoneinfo (3.9+).
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo, available_timezones


def tz_convert(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert time between timezones. time_str: 'HH:MM' or 'YYYY-MM-DD HH:MM'."""
    try:
        src = ZoneInfo(from_tz)
        dst = ZoneInfo(to_tz)
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"
    try:
        time_str = time_str.strip()
        if len(time_str) <= 5:
            # HH:MM — use today's date
            today = datetime.date.today()
            parts = time_str.split(":")
            dt = datetime.datetime(today.year, today.month, today.day,
                                   int(parts[0]), int(parts[1]), tzinfo=src)
        else:
            # YYYY-MM-DD HH:MM
            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            dt = dt.replace(tzinfo=src)
        converted = dt.astimezone(dst)
        return converted.strftime("%Y-%m-%d %H:%M %Z")
    except Exception as e:
        return f"parse error: {e}"


def tz_offset(tz_name: str) -> str:
    """Current UTC offset for a timezone (e.g. '+05:30', '-08:00')."""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        offset = now.utcoffset()
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"UTC{sign}{hours:02d}:{minutes:02d}"
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"


def tz_now(tz_name: str) -> str:
    """Current time in a timezone."""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"


def tz_is_dst(tz_name: str) -> str:
    """Whether a timezone is currently in DST."""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        dst = now.dst()
        if dst is None or dst.total_seconds() == 0:
            return f"{tz_name}: not in DST"
        return f"{tz_name}: in DST (offset {dst})"
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"


def tz_diff(tz1: str, tz2: str) -> str:
    """Hour difference between two timezones right now."""
    try:
        z1 = ZoneInfo(tz1)
        z2 = ZoneInfo(tz2)
        now = datetime.datetime.now(datetime.timezone.utc)
        o1 = now.astimezone(z1).utcoffset().total_seconds() / 3600
        o2 = now.astimezone(z2).utcoffset().total_seconds() / 3600
        diff = o2 - o1
        sign = "+" if diff >= 0 else ""
        return f"{tz2} is {sign}{diff:.1f}h from {tz1}"
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"


def tz_list(region: str = "") -> list:
    """List available timezones, optionally filtered by region (e.g. 'America', 'Europe')."""
    all_tz = sorted(available_timezones())
    if region:
        region = region.strip().title()
        return [tz for tz in all_tz if tz.startswith(region)]
    # Return just region prefixes to avoid dumping 500+ entries
    regions = sorted(set(tz.split("/")[0] for tz in all_tz if "/" in tz))
    return regions


def tz_abbreviation(tz_name: str) -> str:
    """Current timezone abbreviation (e.g. 'EST', 'PDT', 'IST')."""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        return now.strftime("%Z")
    except (KeyError, Exception) as e:
        return f"invalid timezone: {e}"


TIMEZONE_FUNCTIONS = {
    "tz_convert": tz_convert,
    "tz_offset": tz_offset,
    "tz_now": tz_now,
    "tz_is_dst": tz_is_dst,
    "tz_diff": tz_diff,
    "tz_list": tz_list,
    "tz_abbreviation": tz_abbreviation,
}
