#!/usr/bin/env python3
"""
collect_injuries.py — Pull injury and suspension data from API-Football.

Available on Pro plan only. Collects:
  - Injuries: player name, type (injury/suspension), expected return
  - Per team per fixture (pre-match injury list)

This is one of the strongest signals in the model:
  - Missing key players (attackers) reduces xG
  - Missing defenders increases conceded goals
  - Full-strength vs injury-hit teams = value bet opportunity

Strategy:
  - Fetch by league + season (returns all injuries for that season)
  - Store with fixture_id and team_id for feature joining
  - Rate weighted by importance: player's avg minutes > 60 = key player

Usage:
  source .env  # needs FOOTBALL_PRO_PLAN=1
  python3 collect_injuries.py --season 2024
  python3 collect_injuries.py --season 2025 --league 39
"""

import os, sys, time, json, sqlite3, argparse, requests
from datetime import timezone, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FOOTBALL_API_KEY, FOOTBALL_API_BASE, FOOTBALL_API_DELAY_SECONDS
from config.leagues  import LEAGUES, SEASONS, CURRENT_SEASON, LEAGUE_BY_ID

import pandas as pd


def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id      INTEGER,
            league_id       INTEGER,
            season          INTEGER,
            team_id         INTEGER,
            team_name       TEXT,
            player_id       INTEGER,
            player_name     TEXT,
            injury_type     TEXT,   -- 'Injury' or 'Suspension'
            injury_reason   TEXT,   -- e.g. 'Knee', 'Yellow Card'
            match_date      TEXT,
            collected_at    TEXT,
            UNIQUE(fixture_id, team_id, player_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_fixture ON injuries(fixture_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_team    ON injuries(team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_league  ON injuries(league_id, season)")
    conn.commit()
    return conn


def api_get(endpoint, params):
    if not FOOTBALL_API_KEY:
        raise EnvironmentError("FOOTBALL_API_KEY not set. Run: source .env")

    url = f"{FOOTBALL_API_BASE}/{endpoint}"
    headers = {
        "x-rapidapi-key":  FOOTBALL_API_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors") and any(data["errors"].values() if isinstance(data["errors"], dict) else data["errors"]):
        raise RuntimeError(f"API error: {data['errors']}")

    remaining = resp.headers.get("X-RateLimit-requests-Remaining", "?")
    print(f"    [API] /{endpoint} → {len(data.get('response', []))} results | {remaining} remaining", flush=True)
    time.sleep(FOOTBALL_API_DELAY_SECONDS)
    return data.get("response", [])


def collect_league_injuries(conn, league_id, season, force=False):
    """Fetch all injuries for a league+season, store per fixture."""
    if not force:
        cur = conn.execute(
            "SELECT COUNT(*) FROM injuries WHERE league_id=? AND season=?",
            (league_id, season))
        if cur.fetchone()[0] > 0:
            print(f"    ↳ Already have injuries for {league_id}/{season} — skipping")
            return 0

    raw = api_get("injuries", {"league": league_id, "season": season})
    if not raw:
        print(f"    ↳ No injury data returned")
        return 0

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for item in raw:
        fixture  = item.get("fixture", {})
        league   = item.get("league", {})
        team     = item.get("team", {})
        player   = item.get("player", {})
        rows.append({
            "fixture_id":    fixture.get("id"),
            "league_id":     league.get("id", league_id),
            "season":        league.get("season", season),
            "team_id":       team.get("id"),
            "team_name":     team.get("name"),
            "player_id":     player.get("id"),
            "player_name":   player.get("name"),
            "injury_type":   player.get("type"),
            "injury_reason": player.get("reason"),
            "match_date":    fixture.get("date", "")[:10] if fixture.get("date") else None,
            "collected_at":  now,
        })

    if rows:
        conn.executemany("""
            INSERT OR IGNORE INTO injuries (
                fixture_id, league_id, season, team_id, team_name,
                player_id, player_name, injury_type, injury_reason,
                match_date, collected_at
            ) VALUES (
                :fixture_id, :league_id, :season, :team_id, :team_name,
                :player_id, :player_name, :injury_type, :injury_reason,
                :match_date, :collected_at
            )
        """, rows)
        conn.commit()
        print(f"    ↳ Saved {len(rows)} injury/suspension records")

    return len(rows)


def get_injury_features(conn, fixture_id, home_team_id, away_team_id):
    """
    Return injury count features for a specific fixture.
    Used during feature engineering.
    """
    def count_for_team(team_id):
        cur = conn.execute("""
            SELECT injury_type, COUNT(*) as cnt
            FROM injuries
            WHERE fixture_id=? AND team_id=?
            GROUP BY injury_type
        """, (fixture_id, team_id))
        rows = {r[0]: r[1] for r in cur.fetchall()}
        return {
            'injuries':    rows.get('Injury', 0),
            'suspensions': rows.get('Suspension', 0),
            'total_out':   sum(rows.values()),
        }

    home = count_for_team(home_team_id)
    away = count_for_team(away_team_id)
    return {
        'home_injuries':    home['injuries'],
        'home_suspensions': home['suspensions'],
        'home_players_out': home['total_out'],
        'away_injuries':    away['injuries'],
        'away_suspensions': away['suspensions'],
        'away_players_out': away['total_out'],
        'injury_advantage': home['total_out'] - away['total_out'],  # negative = home has more out
    }


def export_parquet(conn, parquet_path):
    df = pd.read_sql("SELECT * FROM injuries ORDER BY league_id, season, match_date", conn)
    df.to_parquet(parquet_path, index=False)
    print(f"\n  Exported {len(df):,} injury records → {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect injury/suspension data")
    parser.add_argument("--season",  type=int, default=None)
    parser.add_argument("--league",  type=int, default=None)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--db",      default=None)
    args = parser.parse_args()

    from config.settings import DB_PATH, PARQUET_DIR
    db_path      = args.db or str(DB_PATH)
    parquet_path = str(PARQUET_DIR / "injuries.parquet")

    conn = get_db(db_path)

    leagues_to_collect = [l for l in LEAGUES if not args.league or l["id"] == args.league]
    seasons_to_collect = [args.season] if args.season else (SEASONS + [CURRENT_SEASON])

    print(f"\n── Collecting injuries & suspensions ─────────────────────────")
    print(f"  Leagues: {len(leagues_to_collect)}, Seasons: {seasons_to_collect}")
    print(f"  ⚠️  Pro plan required (free tier has no injury endpoint)\n")

    total = 0
    for league in leagues_to_collect:
        for season in seasons_to_collect:
            print(f"  {league['name']:30s} ({league['country']}) — {season}")
            try:
                n = collect_league_injuries(conn, league["id"], season, args.force)
                total += n
            except Exception as e:
                print(f"    ❌ {e}")
                continue

    print(f"\n  ✅ Done — {total:,} records saved")
    export_parquet(conn, parquet_path)
    conn.close()


if __name__ == "__main__":
    main()
