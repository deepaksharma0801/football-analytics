import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class DownloadTarget:
    url: str
    path: Path


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("statsbomb_download")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def requests_session_with_retries() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_json(session: requests.Session, url: str, dest: Path, logger: logging.Logger) -> None:
    logger.info("Downloading %s", url)
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: {resp.status_code}")
    dest.write_bytes(resp.content)
    # Validate JSON integrity
    try:
        json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at {dest}: {exc}") from exc


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_matches(competition_id: int, season_id: int, session: requests.Session, logger: logging.Logger) -> List[Dict[str, Any]]:
    matches_url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    resp = session.get(matches_url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download matches {matches_url}: {resp.status_code}")
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Matches response not a list")
    logger.info("Matches downloaded: %d", len(data))
    return data


def build_targets(matches: Iterable[Dict[str, Any]], base_dir: Path) -> List[DownloadTarget]:
    targets: List[DownloadTarget] = []
    for match in matches:
        match_id = match.get("match_id")
        if match_id is None:
            continue
        events_url = f"{BASE_URL}/events/{match_id}.json"
        lineups_url = f"{BASE_URL}/lineups/{match_id}.json"
        targets.append(DownloadTarget(events_url, base_dir / "events" / f"{match_id}.json"))
        targets.append(DownloadTarget(lineups_url, base_dir / "lineups" / f"{match_id}.json"))
    return targets


def create_manifest(files: List[Path], manifest_path: Path) -> None:
    manifest: Dict[str, str] = {}
    for f in files:
        manifest[str(f)] = sha256_file(f)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download StatsBomb Open Data events and lineups")
    parser.add_argument("--competition", type=int, required=True, help="Competition ID")
    parser.add_argument("--season", type=int, required=True, help="Season ID")
    parser.add_argument("--output-root", type=str, default="data/raw/statsbomb", help="Output root directory")
    parser.add_argument("--max-matches", type=int, default=None, help="Limit number of matches")
    args = parser.parse_args()

    logger = setup_logger()
    session = requests_session_with_retries()

    base_dir = Path(args.output_root) / str(args.competition) / str(args.season)
    ensure_dir(base_dir / "events")
    ensure_dir(base_dir / "lineups")

    # Download matches list (used for events/lineups)
    matches = get_matches(args.competition, args.season, session, logger)
    if args.max_matches:
        matches = matches[: args.max_matches]

    targets = build_targets(matches, base_dir)
    downloaded: List[Path] = []

    for target in targets:
        try:
            if target.path.exists():
                logger.info("Already exists, skipping: %s", target.path)
                downloaded.append(target.path)
                continue
            ensure_dir(target.path.parent)
            download_json(session, target.url, target.path, logger)
            downloaded.append(target.path)
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.error("Failed downloading %s: %s", target.url, exc)
            raise

    manifest_path = base_dir / "manifest.json"
    create_manifest(downloaded, manifest_path)
    logger.info("Manifest written: %s", manifest_path)


if __name__ == "__main__":
    main()
