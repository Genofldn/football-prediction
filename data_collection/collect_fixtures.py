#!/usr/bin/env python3
"""
collect_fixtures.py — Pull historical match results for all 24 leagues.

API-Football free tier: 100 requests/day.
Strategy:
  - 1 request per league per season = 120 total for 5 seasons
  - Run over 2 days (60/day), well within the limit
  - After initial load: ~24 requests/day to keep current season updated

Stores to SQLite: data/football.db (table: fixtures)
Also exports to parquet: data/parquet/fixtures.parquet

Usage:
  export FOOTBALL_API_KEY="your_key_here"
  python3 collect_fixtures.py               # collect all leagues, all seasons
  python3 collect_fixtures.py --update      # only update current season (24 requests)
  python3 collect_fixtures.py --league 39   # single league, all seasons
"""

import os, sys, time, json, sqlite3, argparse, requests
from datetime import timezone, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FOOTBALL_API_KEY, FOOTBALL_API_BASE, FOOTBALL_API_DELAY_SECONDS
from config.leagues  import LEAGUES, SEASONS, CURRENT_SEASON, LEAGUE_BY_ID

import pandas as pd


# ── Database setup ─────────────────────────────────────────────────────────────

def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id      INTEGER PRIMARY KEY,
            league_id       INTEGER NOT NULL,
            league_name     TEXT,
            country         TEXT,
            division        INTEGER,
            season          INTEGER NOT NULL,
            date            TEXT,
            timestamp       INTEGER,
            status          TEXT,
            home_team_id    INTEGER,
            home_team       TEXT,
            away_team_id    INTEGER,
            away_team       TEXT,
            home_goals      INTEGER,
            away_goals      INTEGER,
            home_ht         INTEGER,
            away_ht         INTEGER,
            result          TEXT,     -- H / D / A (home win / draw / away win)
            venue           TEXT,
            referee         TEXT,
            raw_json        TEXT,
            collected_at    TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_league_season ON fixtures(league_id, season)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON fixtures(date)")
    conn.commit()
    return conn


# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(endpoint, params):
    """Single API-Football request with rate-limit delay."""
    if not FOOTBALL_API_KEY:
        raise EnvironmentError(
            "FOOTBALL_API_KEY not set. Run: export FOOTBALL_API_KEY='your_key'")

    url = f"{FOOTBALL_API_BASE}/{endpoint}"
    headers = {
        "x-rapidapi-key":  FOOTBALL_API_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # API-Football wraps responses in {"response": [...], "errors": {...}}
    if data.get("errors"):
        errors = data["errors"]
        if errors:
            raise RuntimeError(f"API error: {errors}")

    remaining = resp.headers.get("X-RateLimit-requests-Remaining", "?")
    print(f"    [API] /{endpoint} → {len(data.get('response', []))} results "
          f"| {remaining} requests remaining today", flush=True)

    time.sleep(FOOTBALL_API_DELAY_SECONDS)
    return data.get("response", [])


# ── Parse fixture ──────────────────────────────────────────────────────────────

def parse_fixture(fix, league_id, season):
    """Extract flat fields from a raw API-Football fixture object."""
    f   = fix.get("fixture", {})
    lg  = fix.get("league",  {})
    t   = fix.get("teams",   {})
    g   = fix.get("goals",   {})
    sc  = fix.get("score",   {})

    home_goals = g.get("home")
    away_goals = g.get("away")

    # Result from home team's perspective
    if home_goals is not None and away_goals is not None:
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"
    else:
        result = None  # match not yet played or abandoned

    halftime = sc.get("halftime", {}) or {}

    info = LEAGUE_BY_ID.get(league_id, {})

    return {
        "fixture_id":    f.get("id"),
        "league_id":     league_id,
        "league_name":   lg.get("name") or info.get("name"),
        "country":       lg.get("country") or info.get("country"),
        "division":      info.get("div"),
        "season":        season,
        "date":          f.get("date", "")[:10] if f.get("date") else None,
        "timestamp":     f.get("timestamp"),
        "status":        f.get("status", {}).get("short"),
        "home_team_id":  t.get("home", {}).get("id"),
        "home_team":     t.get("home", {}).get("name"),
        "away_team_id":  t.get("away", {}).get("id"),
        "away_team":     t.get("away", {}).get("name"),
        "home_goals":    home_goals,
        "away_goals":    away_goals,
        "home_ht":       halftime.get("home"),
        "away_ht":       halftime.get("away"),
        "result":        result,
        "venue":         f.get("venue", {}).get("name") if f.get("venue") else None,
        "referee":       f.get("referee"),
        "raw_json":      json.dumps(fix),
        "collected_at":  datetime.now(timezone.utc).isoformat(),
    }


# ── Main collection ────────────────────────────────────────────────────────────

def collect_league_season(conn, league_id, season, force=False):
    """
    Fetch all fixtures for one league + season.
    Skips if already collected (unless force=True).
    Returns count of new records inserted.
    """
    # Check if already collected
    if not force:
        cur = conn.execute(
            "SELECT COUNT(*) FROM fixtures WHERE league_id=? AND season=?",
            (league_id, season))
        existing = cur.fetchone()[0]
        if existing > 0:
            print(f"    ↳ Already have {existing} fixtures — skipping "
                  f"(use --force to re-fetch)")
            return 0

    fixtures_raw = api_get("fixtures", {"league": league_id, "season": season})

    if not fixtures_raw:
        print(f"    ↳ No data returned")
        return 0

    rows = [parse_fixture(f, league_id, season) for f in fixtures_raw]

    # Upsert (replace on conflict with fixture_id primary key)
    conn.executemany("""
        INSERT OR REPLACE INTO fixtures (
            fixture_id, league_id, league_name, country, division, season,
            date, timestamp, status, home_team_id, home_team,
            away_team_id, away_team, home_goals, away_goals, home_ht, away_ht,
            result, venue, referee, raw_json, collected_at
        ) VALUES (
            :fixture_id, :league_id, :league_name, :country, :division, :season,
            :date, :timestamp, :status, :home_team_id, :home_team,
            :away_team_id, :away_team, :home_goals, :away_goals, :home_ht, :away_ht,
            :result, :venue, :referee, :raw_json, :collected_at
        )
    """, rows)
    conn.commit()

    finished = sum(1 for r in rows if r["result"] is not None)
    print(f"    ↳ Saved {len(rows)} fixtures ({finished} completed)")
    return len(rows)


def export_parquet(conn, parquet_path):
    """Export fixtures table to parquet (all fixtures — completed + scheduled)."""
    df = pd.read_sql("SELECT * FROM fixtures ORDER BY date", conn)
    df.to_parquet(parquet_path, index=False)
    n_done    = df['result'].notna().sum()
    n_sched   = df['result'].isna().sum()
    print(f"\n  Exported {len(df):,} fixtures → {parquet_path}")
    print(f"  ({n_done:,} completed + {n_sched:,} scheduled/upcoming)")
    return df


def print_summary(conn):
    """Print a summary of what's in the database."""
    print("\n  ── Database summary ──────────────────────────────────────")
    rows = conn.execute("""
        SELECT league_name, country, division, season,
               COUNT(*) as total,
               SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) as completed
        FROM fixtures
        GROUP BY league_id, season
        ORDER BY country, division, season
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:30s} {r[3]}  {r[5]:>3}/{r[4]:>3} completed")
    total = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
    done  = conn.execute(
        "SELECT COUNT(*) FROM fixtures WHERE result IS NOT NULL").fetchone()[0]
    print(f"\n  Total: {total:,} fixtures | {done:,} completed")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect football fixtures from API-Football")
    parser.add_argument("--update",  action="store_true",
                        help="Only update current season (24 requests, use daily)")
    parser.add_argument("--league",  type=int, default=None,
                        help="Only collect this league ID")
    parser.add_argument("--season",  type=int, default=None,
                        help="Only collect this season")
    parser.add_argument("--force",   action="store_true",
                        help="Re-fetch even if data already exists")
    parser.add_argument("--db",      default=None,
                        help="SQLite database path (default: data/football.db)")
    args = parser.parse_args()

    from config.settings import DB_PATH, PARQUET_DIR
    db_path     = args.db or str(DB_PATH)
    parquet_path = str(PARQUET_DIR / "fixtures.parquet")

    conn = get_db(db_path)

    # Determine which leagues and seasons to collect
    if args.league:
        leagues_to_collect = [l for l in LEAGUES if l["id"] == args.league]
        if not leagues_to_collect:
            print(f"❌ League ID {args.league} not in config")
            sys.exit(1)
    else:
        leagues_to_collect = LEAGUES

    if args.update:
        seasons_to_collect = [CURRENT_SEASON]
        print(f"\n── Daily update: current season {CURRENT_SEASON} "
              f"({len(leagues_to_collect)} leagues = {len(leagues_to_collect)} requests) ──")
    elif args.season:
        seasons_to_collect = [args.season]
    else:
        seasons_to_collect = SEASONS + [CURRENT_SEASON]

    # Estimate request count
    total_requests = len(leagues_to_collect) * len(seasons_to_collect)
    print(f"\n── Collecting fixtures ──────────────────────────────────────")
    print(f"  Leagues:  {len(leagues_to_collect)}")
    print(f"  Seasons:  {seasons_to_collect}")
    print(f"  Requests: ~{total_requests} (free tier: 100/day)")
    if total_requests > 90:
        print(f"  ⚠️  {total_requests} requests > 90 daily limit.")
        print(f"     Split across multiple days or use --season to collect one year at a time.")
        user_in = input("  Continue anyway? [y/N] ").strip().lower()
        if user_in != 'y':
            print("  Aborted.")
            sys.exit(0)
    print()

    total_new = 0
    for league in leagues_to_collect:
        for season in seasons_to_collect:
            print(f"  {league['name']:30s} ({league['country']}) — {season}")
            try:
                new = collect_league_season(conn, league["id"], season, force=args.force)
                total_new += new
            except Exception as e:
                print(f"    ❌ Error: {e}")
                # Don't stop — continue with next league
                continue

    print(f"\n  ✅ Done — {total_new:,} new records added")
    print_summary(conn)
    export_parquet(conn, parquet_path)
    conn.close()


if __name__ == "__main__":
    main()
