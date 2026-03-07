import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("statsbomb_normalize")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_event_files(input_root: Path) -> List[Path]:
    return sorted(input_root.glob("*/**/events/*.json"))


def list_lineup_files(input_root: Path) -> List[Path]:
    return sorted(input_root.glob("*/**/lineups/*.json"))


def get_xy(loc: Optional[List[float]]) -> Tuple[Optional[float], Optional[float]]:
    if loc and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None, None


def summarize_freeze_frame(event: Dict[str, Any]) -> Dict[str, Any]:
    shot = event.get("shot") or {}
    freeze = shot.get("freeze_frame") or []
    loc = event.get("location") or []
    x, y = get_xy(loc)

    def dist(p: List[float]) -> Optional[float]:
        if x is None or y is None or len(p) < 2:
            return None
        return ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5

    teammates = 0
    opponents = 0
    gk = 0
    opp_dists: List[float] = []
    gk_dist: Optional[float] = None

    for obj in freeze:
        if obj.get("teammate"):
            teammates += 1
        else:
            opponents += 1
        if obj.get("position", {}).get("name") == "Goalkeeper":
            gk += 1
            d = dist(obj.get("location") or [])
            if d is not None:
                gk_dist = d
        if not obj.get("teammate"):
            d = dist(obj.get("location") or [])
            if d is not None:
                opp_dists.append(d)

    mean_opp_dist = sum(opp_dists) / len(opp_dists) if opp_dists else None

    return {
        "freeze_teammates": teammates,
        "freeze_opponents": opponents,
        "freeze_goalkeepers": gk,
        "freeze_mean_opponent_dist": mean_opp_dist,
        "freeze_goalkeeper_dist": gk_dist,
    }


def parse_events(event_files: Iterable[Path], logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in event_files:
        file_match_id = int(path.stem)
        data = load_json(path)
        if not isinstance(data, list):
            logger.warning("Events file not list: %s", path)
            continue
        for ev in data:
            loc = ev.get("location")
            x, y = get_xy(loc)
            end_loc = None
            if ev.get("pass") and ev["pass"].get("end_location"):
                end_loc = ev["pass"].get("end_location")
            elif ev.get("carry") and ev["carry"].get("end_location"):
                end_loc = ev["carry"].get("end_location")
            elif ev.get("shot") and ev["shot"].get("end_location"):
                end_loc = ev["shot"].get("end_location")

            end_x, end_y = get_xy(end_loc)
            body_part = None
            if ev.get("shot"):
                body_part = ev["shot"].get("body_part", {}).get("name")
            elif ev.get("pass"):
                body_part = ev["pass"].get("body_part", {}).get("name")

            shot_xg = ev.get("shot", {}).get("statsbomb_xg")
            shot_type = ev.get("shot", {}).get("type", {}).get("name")
            assist_type = ev.get("shot", {}).get("assist_type", {}).get("name")
            shot_outcome = ev.get("shot", {}).get("outcome", {}).get("name")
            pass_length = ev.get("pass", {}).get("length")
            pass_angle = ev.get("pass", {}).get("angle")
            pass_assisted_shot_id = ev.get("pass", {}).get("assisted_shot_id")
            pass_goal_assist = ev.get("pass", {}).get("goal_assist")
            pressure_flag = ev.get("type", {}).get("name") == "Pressure"

            freeze_features = summarize_freeze_frame(ev)

            rows.append(
                {
                    "event_id": ev.get("id"),
                    "match_id": ev.get("match_id") or file_match_id,
                    "minute": ev.get("minute"),
                    "second": ev.get("second"),
                    "period": ev.get("period"),
                    "team_id": ev.get("team", {}).get("id"),
                    "player_id": ev.get("player", {}).get("id"),
                    "event_type": ev.get("type", {}).get("name"),
                    "x": x,
                    "y": y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "body_part": body_part,
                    "shot_statsbomb_xg": shot_xg,
                    "shot_type": shot_type,
                    "assist_type": assist_type,
                    "shot_outcome": shot_outcome,
                    "pass_length": pass_length,
                    "pass_angle": pass_angle,
                    "pass_assisted_shot_id": pass_assisted_shot_id,
                    "pass_goal_assist": pass_goal_assist,
                    "possession": ev.get("possession"),
                    "pressure_flag": pressure_flag,
                    "freeze_frame_features": json.dumps(freeze_features),
                }
            )
    df = pd.DataFrame(rows)
    return df


def parse_lineups(lineup_files: Iterable[Path], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lineup_rows: List[Dict[str, Any]] = []
    teams_rows: List[Dict[str, Any]] = []
    players_rows: List[Dict[str, Any]] = []

    for path in lineup_files:
        match_id = int(path.stem)
        data = load_json(path)
        if not isinstance(data, list):
            logger.warning("Lineups file not list: %s", path)
            continue
        for team in data:
            team_id = team.get("team_id") or team.get("team", {}).get("id")
            team_name = team.get("team_name") or team.get("team", {}).get("name")
            teams_rows.append({"team_id": team_id, "team_name": team_name})
            for player in team.get("lineup", []):
                player_id = player.get("player_id") or player.get("player", {}).get("id")
                player_name = player.get("player_name") or player.get("player", {}).get("name")
                position = player.get("position", {}) or {}
                players_rows.append({"player_id": player_id, "player_name": player_name})
                lineup_rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_id": player_id,
                        "position_id": position.get("id"),
                        "position_name": position.get("name"),
                        "jersey_number": player.get("jersey_number"),
                    }
                )

    teams_df = pd.DataFrame(teams_rows).drop_duplicates(subset=["team_id"]).reset_index(drop=True)
    players_df = pd.DataFrame(players_rows).drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    lineups_df = pd.DataFrame(lineup_rows)
    return teams_df, players_df, lineups_df


def build_matches(events_df: pd.DataFrame, input_root: Path) -> pd.DataFrame:
    matches: List[Dict[str, Any]] = []
    for match_id in events_df["match_id"].dropna().unique():
        match_id_int = int(match_id)
        # infer competition/season from folder structure if possible
        comp = None
        season = None
        # find a directory containing this match file
        event_path = next(input_root.glob(f"*/**/events/{match_id_int}.json"), None)
        if event_path is not None:
            parts = event_path.parts
            # .../statsbomb/<competition>/<season>/events/<match_id>.json
            if "statsbomb" in parts:
                idx = parts.index("statsbomb")
                if len(parts) > idx + 2:
                    comp = parts[idx + 1]
                    season = parts[idx + 2]
        matches.append(
            {
                "match_id": match_id_int,
                "competition_id": comp,
                "season_id": season,
            }
        )
    return pd.DataFrame(matches)


def validate_events(events_df: pd.DataFrame) -> None:
    if events_df["match_id"].isna().any():
        raise ValueError("Found null match_id in events")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize StatsBomb events into relational CSVs")
    parser.add_argument("--input-root", type=str, default="data/raw/statsbomb")
    parser.add_argument("--output-root", type=str, default="data/processed")
    args = parser.parse_args()

    logger = setup_logger()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    event_files = list_event_files(input_root)
    lineup_files = list_lineup_files(input_root)

    if not event_files:
        raise FileNotFoundError(f"No event files under {input_root}")

    logger.info("Event files: %d", len(event_files))
    logger.info("Lineup files: %d", len(lineup_files))

    events_df = parse_events(event_files, logger)
    validate_events(events_df)

    teams_df, players_df, lineups_df = parse_lineups(lineup_files, logger)
    matches_df = build_matches(events_df, input_root)

    write_csv(matches_df, output_root / "matches.csv")
    write_csv(events_df, output_root / "events.csv")
    write_csv(players_df, output_root / "players.csv")
    write_csv(teams_df, output_root / "teams.csv")
    write_csv(lineups_df, output_root / "lineups.csv")

    logger.info("Normalization complete")


if __name__ == "__main__":
    main()
