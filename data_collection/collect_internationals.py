#!/usr/bin/env python3
"""
collect_internationals.py — Pull senior national-team match history for World Cup modelling.

Stores results in a SEPARATE `intl_fixtures` table so club models are never affected.
Covers: World Cup + qualifiers (all confederations), Nations League, Euro, Copa America,
AFCON, Asian Cup, Gold Cup, CONCACAF Nations League, and international friendlies.

Usage:
  source .env
  python3 data_collection/collect_internationals.py            # full pull
  python3 data_collection/collect_internationals.py --wc-only  # just World Cup 2026 (refresh)
"""

import os, sys, json, time, argparse, sqlite3, requests
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import FOOTBALL_API_KEY, FOOTBALL_API_BASE

DELAY = float(os.environ.get("FOOTBALL_API_DELAY_SECONDS", "1"))
DB_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "football.db")

# competition_id : (name, neutral_venue?, [seasons])
#   neutral=True  → final-tournament matches played at neutral host venues (no home edge)
#   neutral=False → qualifiers / friendlies / nations league (home/away meaningful)
COMPETITIONS = {
    1:   ("FIFA World Cup",                 True,  [2018, 2022, 2026]),
    4:   ("UEFA Euro",                      True,  [2020, 2024]),
    9:   ("Copa America",                   True,  [2019, 2021, 2024]),
    6:   ("Africa Cup of Nations",          True,  [2019, 2021, 2023, 2025]),
    7:   ("AFC Asian Cup",                  True,  [2019, 2023]),
    22:  ("CONCACAF Gold Cup",              True,  [2019, 2021, 2023, 2025]),
    5:   ("UEFA Nations League",            False, [2018, 2020, 2022, 2024, 2026]),
    536: ("CONCACAF Nations League",        False, [2020, 2022, 2023, 2024]),
    32:  ("WC Qualification Europe",        False, [2020, 2024]),
    34:  ("WC Qualification S.America",     False, [2022, 2026]),
    29:  ("WC Qualification Africa",        False, [2022, 2023]),
    30:  ("WC Qualification Asia",          False, [2022, 2026]),
    31:  ("WC Qualification CONCACAF",      False, [2022, 2026]),
    33:  ("WC Qualification Oceania",       False, [2026]),
    37:  ("WC Qualification Intercont.",    False, [2026]),
    10:  ("International Friendlies",       False, [2021, 2022, 2023, 2024, 2025, 2026]),
}


def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intl_fixtures (
            fixture_id    INTEGER PRIMARY KEY,
            league_id     INTEGER NOT NULL,
            league_name   TEXT,
            season        INTEGER NOT NULL,
            round         TEXT,
            date          TEXT,
            timestamp     INTEGER,
            status        TEXT,
            neutral       INTEGER,
            home_team_id  INTEGER,
            home_team     TEXT,
            away_team_id  INTEGER,
            away_team     TEXT,
            home_goals    INTEGER,
            away_goals    INTEGER,
            result        TEXT,
            collected_at  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_intl_date ON intl_fixtures(date)")
    conn.commit()
    return conn


def api_get(endpoint, params):
    url = f"{FOOTBALL_API_BASE}/{endpoint}"
    headers = {"x-rapidapi-key": FOOTBALL_API_KEY, "x-rapidapi-host": "v3.football.api-sports.io"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"API error: {data['errors']}")
    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    time.sleep(DELAY)
    return data.get("response", []), remaining


def parse(fix, league_id, season, neutral):
    f = fix.get("fixture", {}); lg = fix.get("league", {})
    t = fix.get("teams", {});   g = fix.get("goals", {})
    hg, ag = g.get("home"), g.get("away")
    if hg is not None and ag is not None:
        result = "H" if hg > ag else "A" if hg < ag else "D"
    else:
        result = None
    return {
        "fixture_id": f.get("id"), "league_id": league_id,
        "league_name": lg.get("name"), "season": season, "round": lg.get("round"),
        "date": f.get("date", "")[:10] if f.get("date") else None,
        "timestamp": f.get("timestamp"), "status": f.get("status", {}).get("short"),
        "neutral": 1 if neutral else 0,
        "home_team_id": t.get("home", {}).get("id"), "home_team": t.get("home", {}).get("name"),
        "away_team_id": t.get("away", {}).get("id"), "away_team": t.get("away", {}).get("name"),
        "home_goals": hg, "away_goals": ag, "result": result,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def store(conn, rows):
    conn.executemany("""
        INSERT OR REPLACE INTO intl_fixtures
        (fixture_id, league_id, league_name, season, round, date, timestamp, status,
         neutral, home_team_id, home_team, away_team_id, away_team, home_goals, away_goals,
         result, collected_at)
        VALUES (:fixture_id,:league_id,:league_name,:season,:round,:date,:timestamp,:status,
         :neutral,:home_team_id,:home_team,:away_team_id,:away_team,:home_goals,:away_goals,
         :result,:collected_at)
    """, rows)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wc-only", action="store_true", help="Only refresh World Cup 2026")
    ap.add_argument("--db", default=DB_DEFAULT)
    args = ap.parse_args()

    if not FOOTBALL_API_KEY:
        print("❌ FOOTBALL_API_KEY not set — run: source .env"); sys.exit(1)

    conn = get_db(args.db)
    comps = {1: ("FIFA World Cup", True, [2026])} if args.wc_only else COMPETITIONS

    total_rows = total_done = 0
    for lid, (name, neutral, seasons) in comps.items():
        for season in seasons:
            try:
                resp, remaining = api_get("fixtures", {"league": lid, "season": season})
            except Exception as e:
                print(f"  ⚠️  {name} {season}: {e}")
                continue
            rows = [parse(f, lid, season, neutral) for f in resp]
            if rows:
                store(conn, rows)
            done = sum(1 for r in rows if r["result"] is not None)
            total_rows += len(rows); total_done += done
            print(f"  {name:30s} {season}  {len(rows):4d} fixtures ({done} played) | {remaining} req left")

    n = conn.execute("SELECT COUNT(*) FROM intl_fixtures").fetchone()[0]
    n_done = conn.execute("SELECT COUNT(*) FROM intl_fixtures WHERE result IS NOT NULL").fetchone()[0]
    teams = conn.execute("SELECT COUNT(DISTINCT home_team_id) FROM intl_fixtures").fetchone()[0]
    print(f"\n✅ intl_fixtures: {n} total ({n_done} played), ~{teams} teams. Added {total_rows} this run.")
    conn.close()


if __name__ == "__main__":
    main()
