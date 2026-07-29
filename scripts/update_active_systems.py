#!/usr/bin/env python3
"""
Build data/active_systems.json from live NOAA/NHC and NOAA/OSPO sources.

Sources
-------
1. NHC CurrentStorms.json
   - Atlantic, Eastern Pacific, and Central Pacific active cyclones.
2. NOAA OSPO Tropical Bulletins
   - Recent bulletins for additional basins, including the Western Pacific
     and North Indian Ocean when listed.

Output schema is compatible with main.py:
{
  "updated_at_utc": "...",
  "systems": [
    {
      "name": "...",
      "storm_id": "...",
      "latitude": 12.3,
      "longitude": 132.4,
      "time": "...",
      "status": "TS",
      "source": "NHC_CURRENT_STORMS"
    }
  ]
}

Notes
-----
- OSPO states that its posted positions/intensities may differ from official
  warning-centre information. The source field is therefore preserved.
- This script keeps only the newest position for each storm ID.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
OSPO_BULLETINS_URL = (
    "https://www.ospo.noaa.gov/products/ocean/tropical/bulletins.html"
)

ACTIVE_STATUSES = {
    "TD", "TS", "STS", "TC", "TY", "HU", "CY", "ST", "SS",
    "SD", "DB", "PTC",
}

NHC_STATUS_MAP = {
    "HU": "HU",
    "TS": "TS",
    "TD": "TD",
    "ST": "ST",
    "SS": "SS",
    "SD": "SD",
    "PT": "PTC",
    "PTC": "PTC",
}

# Approximate Dvorak current-intensity classification to operational category.
DVORAK_STATUS = (
    (5.0, "TY"),
    (3.5, "TS"),
    (2.0, "TD"),
    (0.0, "DB"),
)


@dataclass(frozen=True)
class ActiveSystem:
    name: str
    storm_id: str
    latitude: float
    longitude: float
    time: str
    status: str
    source: str
    intensity_kt: Optional[float] = None
    pressure_hpa: Optional[float] = None
    basin: Optional[str] = None
    bulletin_url: Optional[str] = None


class LinkCollector(HTMLParser):
    """Collect links and their visible text using only the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._href: Optional[str] = None
        self._text_parts: List[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        if tag.lower() != "a":
            return
        self._href = dict(attrs).get("href")
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = " ".join("".join(self._text_parts).split())
            self.links.append((self._href, text))
            self._href = None
            self._text_parts = []


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_datetime(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def parse_coordinate(value: str) -> float:
    """
    Convert coordinates such as 13.4N, 177.0E, 109.4W, or -30.2.
    """
    text = value.strip().upper()
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([NSEW])?", text)
    if not match:
        raise ValueError(f"Invalid coordinate: {value!r}")

    number = float(match.group(1))
    hemisphere = match.group(2)

    if hemisphere in {"S", "W"}:
        number = -abs(number)
    elif hemisphere in {"N", "E"}:
        number = abs(number)

    return number


def build_session(insecure_ssl: bool) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": (
                "TC-Genesis-Research/1.0 "
                "(active-system post-classification updater)"
            ),
            "Accept": "application/json,text/html,text/plain,*/*",
        }
    )
    session.verify = not insecure_ssl

    if insecure_ssl:
        requests.packages.urllib3.disable_warnings(  # type: ignore[attr-defined]
            requests.packages.urllib3.exceptions.InsecureRequestWarning  # type: ignore[attr-defined]
        )

    return session


def get_text(
    session: requests.Session,
    url: str,
    *,
    timeout: Tuple[int, int] = (20, 90),
) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_nhc_systems(
    session: requests.Session,
    *,
    now: datetime,
    max_age_hours: float,
) -> List[ActiveSystem]:
    logging.info("Fetching NHC active storms: %s", NHC_CURRENT_STORMS_URL)
    response = session.get(NHC_CURRENT_STORMS_URL, timeout=(20, 90))
    response.raise_for_status()
    payload = response.json()

    storms = payload.get("activeStorms", [])
    if not isinstance(storms, list):
        raise ValueError("NHC response does not contain an activeStorms list.")

    output: List[ActiveSystem] = []

    for storm in storms:
        try:
            position_time = parse_iso_datetime(storm["lastUpdate"])
            age_hours = (now - position_time).total_seconds() / 3600.0
            if age_hours < -2.0 or age_hours > max_age_hours:
                continue

            latitude = float(
                storm.get(
                    "latitudeNumeric",
                    parse_coordinate(str(storm["latitude"])),
                )
            )
            longitude = float(
                storm.get(
                    "longitudeNumeric",
                    parse_coordinate(str(storm["longitude"])),
                )
            )

            raw_status = str(storm.get("classification", "TC")).upper()
            status = NHC_STATUS_MAP.get(raw_status, raw_status)
            if status not in ACTIVE_STATUSES:
                status = "TC"

            output.append(
                ActiveSystem(
                    name=str(storm.get("name") or "UNNAMED").upper(),
                    storm_id=str(storm.get("id") or "UNKNOWN").upper(),
                    latitude=latitude,
                    longitude=longitude,
                    time=iso_z(position_time),
                    status=status,
                    source="NHC_CURRENT_STORMS",
                    intensity_kt=_optional_float(storm.get("intensity")),
                    pressure_hpa=_optional_float(storm.get("pressure")),
                    basin=_nhc_basin_from_id(str(storm.get("id", ""))),
                    bulletin_url=_nested_url(storm, "publicAdvisory"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logging.warning("Skipping malformed NHC storm entry: %s", exc)

    logging.info("NHC returned %d recent active systems.", len(output))
    return output


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_url(mapping: Dict[str, Any], key: str) -> Optional[str]:
    value = mapping.get(key)
    if isinstance(value, dict):
        url = value.get("url")
        return str(url) if url else None
    return None


def _nhc_basin_from_id(storm_id: str) -> Optional[str]:
    prefix = storm_id[:2].upper()
    return {
        "AL": "North Atlantic",
        "EP": "Eastern Pacific",
        "CP": "Central Pacific",
    }.get(prefix)


def find_ospo_bulletin_links(
    session: requests.Session,
    *,
    now: datetime,
    max_age_hours: float,
) -> List[Tuple[str, str, datetime, Optional[str]]]:
    """
    Return newest bulletin link candidates:
    (storm_id, url, nominal_time, basin).
    """
    logging.info("Fetching NOAA OSPO bulletin index: %s", OSPO_BULLETINS_URL)
    html = get_text(session, OSPO_BULLETINS_URL)
    parser = LinkCollector()
    parser.feed(html)

    candidates: List[Tuple[str, str, datetime, Optional[str]]] = []
    seen_urls: set[str] = set()

    # Expected visible text resembles "0530 UTC"; storm ID and date are often
    # encoded in the bulletin URL: .../20260727053012W.html
    url_pattern = re.compile(
        r"/(?P<basin>[^/]+)/(?P<stamp>\d{12})(?P<storm>[A-Z0-9]+)\.html$",
        re.IGNORECASE,
    )

    for href, _link_text in parser.links:
        full_url = urljoin(OSPO_BULLETINS_URL, href)
        match = url_pattern.search(full_url)
        if not match or full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        try:
            nominal_time = datetime.strptime(
                match.group("stamp"), "%Y%m%d%H%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        age_hours = (now - nominal_time).total_seconds() / 3600.0
        if age_hours < -2.0 or age_hours > max_age_hours:
            continue

        candidates.append(
            (
                match.group("storm").upper(),
                full_url,
                nominal_time,
                match.group("basin").lower(),
            )
        )

    # Newest bulletin first; deduplication by storm happens after parsing.
    candidates.sort(key=lambda item: item[2], reverse=True)
    logging.info(
        "Found %d recent NOAA OSPO bulletin links.", len(candidates)
    )
    return candidates


def dvorak_status(text: str) -> str:
    match = re.search(r"\bT(\d+(?:\.\d+)?)\s*/", text, re.IGNORECASE)
    if not match:
        return "TC"
    t_number = float(match.group(1))
    for threshold, status in DVORAK_STATUS:
        if t_number >= threshold:
            return status
    return "DB"


def parse_ospo_bulletin(
    text: str,
    *,
    fallback_storm_id: str,
    fallback_time: datetime,
    source_url: str,
    basin_code: Optional[str],
) -> ActiveSystem:
    """
    Parse NOAA OSPO Dvorak bulletin fields A-D and F.

    Typical form:
      A.  12W (NONAME)
      B.  27/0530Z
      C.  13.4N
      D.  177.0E
      F.  T2.5/2.5
    """
    normalized = "\n".join(line.strip() for line in text.splitlines())

    id_name = re.search(
        r"(?m)^\s*A\.\s+([A-Z0-9]+)(?:\s+\(([^)]+)\))?",
        normalized,
        re.IGNORECASE,
    )
    latitude_match = re.search(
        r"(?m)^\s*C\.\s+([+-]?\d+(?:\.\d+)?\s*[NS]?)",
        normalized,
        re.IGNORECASE,
    )
    longitude_match = re.search(
        r"(?m)^\s*D\.\s+([+-]?\d+(?:\.\d+)?\s*[EW]?)",
        normalized,
        re.IGNORECASE,
    )
    time_match = re.search(
        r"(?m)^\s*B\.\s+(\d{1,2})/(\d{4})Z",
        normalized,
        re.IGNORECASE,
    )

    if not latitude_match or not longitude_match:
        raise ValueError("OSPO bulletin is missing position fields C/D.")

    storm_id = (
        id_name.group(1).upper()
        if id_name
        else fallback_storm_id.upper()
    )
    raw_name = id_name.group(2).strip().upper() if id_name and id_name.group(2) else ""
    name = raw_name if raw_name not in {"", "NONAME", "NO NAME"} else storm_id

    bulletin_time = fallback_time
    if time_match:
        day = int(time_match.group(1))
        hour = int(time_match.group(2)[:2])
        minute = int(time_match.group(2)[2:])

        # Start from fallback month/year and adjust across month boundaries.
        candidate = fallback_time.replace(
            day=1, hour=hour, minute=minute, second=0, microsecond=0
        )
        month_candidates = [
            candidate - timedelta(days=3),
            candidate,
            candidate + timedelta(days=35),
        ]
        valid: List[datetime] = []
        for month_seed in month_candidates:
            try:
                valid.append(
                    month_seed.replace(day=day, hour=hour, minute=minute)
                )
            except ValueError:
                pass
        if valid:
            bulletin_time = min(
                valid,
                key=lambda item: abs(
                    (item - fallback_time).total_seconds()
                ),
            )

    return ActiveSystem(
        name=name,
        storm_id=storm_id,
        latitude=parse_coordinate(latitude_match.group(1)),
        longitude=parse_coordinate(longitude_match.group(1)),
        time=iso_z(bulletin_time),
        status=dvorak_status(normalized),
        source="NOAA_OSPO_DVORAK_BULLETIN",
        intensity_kt=None,
        pressure_hpa=None,
        basin=_ospo_basin_name(basin_code),
        bulletin_url=source_url,
    )


def _ospo_basin_name(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    return {
        "wpac": "Western Pacific",
        "nind": "North Indian Ocean",
        "spac": "South Pacific",
        "epac": "Eastern Pacific",
        "atl": "North Atlantic",
    }.get(code.lower(), code)


def fetch_ospo_systems(
    session: requests.Session,
    *,
    now: datetime,
    max_age_hours: float,
) -> List[ActiveSystem]:
    links = find_ospo_bulletin_links(
        session,
        now=now,
        max_age_hours=max_age_hours,
    )

    newest_by_id: Dict[str, ActiveSystem] = {}

    for fallback_id, url, nominal_time, basin_code in links:
        # We only need the newest successful bulletin per ID.
        if fallback_id in newest_by_id:
            continue
        try:
            text = get_text(session, url)
            system = parse_ospo_bulletin(
                text,
                fallback_storm_id=fallback_id,
                fallback_time=nominal_time,
                source_url=url,
                basin_code=basin_code,
            )

            system_time = parse_iso_datetime(system.time)
            age_hours = (now - system_time).total_seconds() / 3600.0
            if -2.0 <= age_hours <= max_age_hours:
                newest_by_id[system.storm_id] = system
        except (requests.RequestException, ValueError) as exc:
            logging.warning("Skipping OSPO bulletin %s: %s", url, exc)

    systems = list(newest_by_id.values())
    logging.info("OSPO returned %d recent systems.", len(systems))
    return systems


def deduplicate_systems(
    systems: Iterable[ActiveSystem],
) -> List[ActiveSystem]:
    """
    Keep the newest record for each source-independent storm key.

    NHC IDs are basin/year IDs, while OSPO IDs are short IDs such as 12W.
    Exact IDs are deduplicated; near-position cross-source duplicates are also
    collapsed when they are within 150 km and 12 hours.
    """
    ordered = sorted(
        systems,
        key=lambda item: parse_iso_datetime(item.time),
        reverse=True,
    )

    retained: List[ActiveSystem] = []
    seen_ids: set[str] = set()

    for candidate in ordered:
        normalized_id = candidate.storm_id.upper()
        if normalized_id in seen_ids:
            continue

        duplicate = False
        candidate_time = parse_iso_datetime(candidate.time)

        for existing in retained:
            existing_time = parse_iso_datetime(existing.time)
            hours = abs(
                (candidate_time - existing_time).total_seconds()
            ) / 3600.0
            distance = haversine_km(
                candidate.latitude,
                candidate.longitude,
                existing.latitude,
                existing.longitude,
            )
            if hours <= 12.0 and distance <= 150.0:
                duplicate = True
                break

        if not duplicate:
            retained.append(candidate)
            seen_ids.add(normalized_id)

    return sorted(
        retained,
        key=lambda item: (
            parse_iso_datetime(item.time),
            item.storm_id,
        ),
        reverse=True,
    )


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    from math import asin, cos, radians, sin, sqrt

    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)

    value = (
        sin(dphi / 2.0) ** 2
        + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * asin(min(1.0, sqrt(value)))


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent active tropical systems from NHC and NOAA OSPO "
            "and write active_systems.json for main.py."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/active_systems.json"),
        help="Output JSON path. Default: data/active_systems.json",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=72.0,
        help="Discard positions older than this. Default: 72",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("nhc", "ospo"),
        default=("nhc", "ospo"),
        help="Sources to query. Default: nhc ospo",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable TLS certificate verification for intercepted proxies.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write output when one source fails, provided another succeeds. "
            "Without this flag, any selected-source failure exits non-zero."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_age_hours <= 0:
        raise ValueError("--max-age-hours must be positive.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    now = utc_now()
    session = build_session(args.insecure_ssl)
    systems: List[ActiveSystem] = []
    source_status: Dict[str, Dict[str, Any]] = {}
    failures = 0

    if "nhc" in args.sources:
        try:
            nhc = fetch_nhc_systems(
                session,
                now=now,
                max_age_hours=args.max_age_hours,
            )
            systems.extend(nhc)
            source_status["nhc"] = {"ok": True, "count": len(nhc)}
        except Exception as exc:
            failures += 1
            source_status["nhc"] = {"ok": False, "error": str(exc)}
            logging.exception("NHC update failed.")

    if "ospo" in args.sources:
        try:
            ospo = fetch_ospo_systems(
                session,
                now=now,
                max_age_hours=args.max_age_hours,
            )
            systems.extend(ospo)
            source_status["ospo"] = {"ok": True, "count": len(ospo)}
        except Exception as exc:
            failures += 1
            source_status["ospo"] = {"ok": False, "error": str(exc)}
            logging.exception("OSPO update failed.")

    selected_source_count = len(args.sources)
    successful_sources = selected_source_count - failures

    if failures and (not args.allow_partial or successful_sources == 0):
        logging.error(
            "Not writing output because %d selected source(s) failed.",
            failures,
        )
        return 2

    deduplicated = deduplicate_systems(systems)

    payload: Dict[str, Any] = {
        "updated_at_utc": iso_z(now),
        "max_age_hours": args.max_age_hours,
        "source_status": source_status,
        "system_count": len(deduplicated),
        "systems": [
            {
                key: value
                for key, value in asdict(system).items()
                if value is not None
            }
            for system in deduplicated
        ],
    }

    write_json_atomic(args.output, payload)

    logging.info("Saved %d systems to %s", len(deduplicated), args.output)
    for system in deduplicated:
        logging.info(
            "%-12s %-8s %6.1f %7.1f %-4s %s",
            system.name,
            system.storm_id,
            system.latitude,
            system.longitude,
            system.status,
            system.source,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)