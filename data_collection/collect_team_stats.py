#!/usr/bin/env python3
"""
collect_team_stats.py — Pull per-team statistics from API-Football.

Fetches team season statistics including:
  - Goals scored/conceded (home/away split)
  - Shots, shots on target
  - Possession average
  - Pass accuracy
  - Clean sheets
  - Win/draw/loss record
  - Form (last 5 matches as WDLWW etc.)
  - Biggest win/loss

API cost: 1 request per team per season.
Strategy:
  - First collect fixtures to know which teams are in each league/season
  - Then collect stats for each unique team
  - ~400 teams across 24 leagues = 400 requests (4 days on free tier)
  - Or: use --big5-only for ~100 teams (1 day)

Usage:
  source .env
  python3 collect_team_stats.py --update            # current season only
  python3 collect_team_stats.py --league 39         # single league
  python3 collect_team_stats.py --big5-only         # 10 Big 5 leagues only
  python3 collect_team_stats.py --season 2024       # specific season
"""

import os, sys, time, json, sqlite3, argparse, requests
from datetime import timezone, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FOOTBALL_API_KEY, FOOTBALL_API_BASE, FOOTBALL_API_DELAY_SECONDS
from config.leagues  import LEAGUES, SEASONS, CURRENT_SEASON, BIG5_IDS, LEAGUE_BY_ID

import pandas as pd


# ── Database setup ─────────────────────────────────────────────────────────────

def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id         INTEGER NOT NULL,
            team_name       TEXT,
            league_id       INTEGER NOT NULL,
            season          INTEGER NOT NULL,
            -- Record
            played_home     INTEGER, played_away     INTEGER, played_total    INTEGER,
            wins_home       INTEGER, wins_away       INTEGER, wins_total      INTEGER,
            draws_home      INTEGER, draws_away      INTEGER, draws_total     INTEGER,
            losses_home     INTEGER, losses_away     INTEGER, losses_total    INTEGER,
            -- Goals
            goals_for_home      REAL, goals_for_away      REAL, goals_for_total      REAL,
            goals_against_home  REAL, goals_against_away  REAL, goals_against_total  REAL,
            goals_for_avg_home  REAL, goals_for_avg_away  REAL, goals_for_avg_total  REAL,
            goals_against_avg_home REAL, goals_against_avg_away REAL, goals_against_avg_total REAL,
            -- Shots
            shots_total     INTEGER, shots_on_target INTEGER,
            -- Possession
            possession_avg  REAL,
            -- Passing
            pass_accuracy   REAL,
            -- Discipline
            yellow_cards    INTEGER, red_cards       INTEGER,
            -- Streak / form
            clean_sheets_home INTEGER, clean_sheets_away INTEGER, clean_sheets_total INTEGER,
            failed_to_score_home INTEGER, failed_to_score_away INTEGER, failed_to_score_total INTEGER,
            -- Biggest results
            biggest_win_home TEXT, biggest_win_away TEXT,
            biggest_loss_home TEXT, biggest_loss_away TEXT,
            -- Streak
            current_streak TEXT,
            -- Raw JSON (for any extra fields)
            raw_json        TEXT,
            collected_at    TEXT,
            UNIQUE(team_id, league_id, season)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_team   ON team_stats(team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_league ON team_stats(league_id, season)")
    conn.commit()
    return conn


# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(endpoint, params):
    """Single API-Football request with rate-limit delay."""
    if not FOOTBALL_API_KEY:
        raise EnvironmentError(
            "FOOTBALL_API_KEY not set. Run: source .env")

    url = f"{FOOTBALL_API_BASE}/{endpoint}"
    headers = {
        "x-rapidapi-key":  FOOTBALL_API_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        errors = data["errors"]
        if errors and any(errors.values() if isinstance(errors, dict) else errors):
            raise RuntimeError(f"API error: {errors}")

    remaining = resp.headers.get("X-RateLimit-requests-Remaining", "?")
    print(f"    [API] /{endpoint} → {remaining} requests remaining today", flush=True)

    time.sleep(FOOTBALL_API_DELAY_SECONDS)
    return data.get("response", [])


# ── Parse team stats ───────────────────────────────────────────────────────────

def _safe(d, *keys, default=None):
    """Safely navigate nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def parse_team_stats(raw, league_id, season):
    """Extract flat fields from API-Football /teams/statistics response."""
    team    = _safe(raw, "team",   default={})
    games   = _safe(raw, "games",  default={})
    goals   = _safe(raw, "goals",  default={})
    shots   = _safe(raw, "shots",  default={})
    passes  = _safe(raw, "passes", default={})
    cards   = _safe(raw, "cards",  default={})
    clean   = _safe(raw, "clean_sheet", default={})
    nogoal  = _safe(raw, "failed_to_score", default={})
    bigwin  = _safe(raw, "biggest", "wins",   default={})
    bigloss = _safe(raw, "biggest", "loses",  default={})
    streak  = _safe(raw, "biggest", "streak", default={})

    # Possession is stored as "45%" — strip the percent
    pos = _safe(games, "matches", "played", "total")  # fallback to played count
    possession_raw = _safe(games, "matches", "played")  # not quite right
    possession_pct = _safe(raw, "possession", "total")
    if isinstance(possession_pct, str):
        possession_pct = float(possession_pct.strip('%'))

    # Pass accuracy is stored as "80%" too
    pass_acc = _safe(passes, "accuracy", "total")
    if isinstance(pass_acc, str) and pass_acc.endswith('%'):
        pass_acc = float(pass_acc.strip('%'))

    return {
        "team_id":    team.get("id"),
        "team_name":  team.get("name"),
        "league_id":  league_id,
        "season":     season,
        # Record
        "played_home":  _safe(games, "played", "home"),
        "played_away":  _safe(games, "played", "away"),
        "played_total": _safe(games, "played", "total"),
        "wins_home":    _safe(games, "wins",   "home"),
        "wins_away":    _safe(games, "wins",   "away"),
        "wins_total":   _safe(games, "wins",   "total"),
        "draws_home":   _safe(games, "draws",  "home"),
        "draws_away":   _safe(games, "draws",  "away"),
        "draws_total":  _safe(games, "draws",  "total"),
        "losses_home":  _safe(games, "loses",  "home"),
        "losses_away":  _safe(games, "loses",  "away"),
        "losses_total": _safe(games, "loses",  "total"),
        # Goals
        "goals_for_home":       _safe(goals, "for",     "total",   "home"),
        "goals_for_away":       _safe(goals, "for",     "total",   "away"),
        "goals_for_total":      _safe(goals, "for",     "total",   "total"),
        "goals_against_home":   _safe(goals, "against", "total",   "home"),
        "goals_against_away":   _safe(goals, "against", "total",   "away"),
        "goals_against_total":  _safe(goals, "against", "total",   "total"),
        "goals_for_avg_home":   _safe(goals, "for",     "average", "home"),
        "goals_for_avg_away":   _safe(goals, "for",     "average", "away"),
        "goals_for_avg_total":  _safe(goals, "for",     "average", "total"),
        "goals_against_avg_home":   _safe(goals, "against", "average", "home"),
        "goals_against_avg_away":   _safe(goals, "against", "average", "away"),
        "goals_against_avg_total":  _safe(goals, "against", "average", "total"),
        # Shots
        "shots_total":      _safe(shots, "total"),
        "shots_on_target":  _safe(shots, "on", "target"),
        # Possession
        "possession_avg": possession_pct,
        # Passing
        "pass_accuracy": pass_acc,
        # Cards
        "yellow_cards": _safe(cards, "yellow", "total"),
        "red_cards":    _safe(cards, "red",    "total"),
        # Clean sheets / failed to score
        "clean_sheets_home":  _safe(clean,  "home"),
        "clean_sheets_away":  _safe(clean,  "away"),
        "clean_sheets_total": _safe(clean,  "total"),
        "failed_to_score_home":  _safe(nogoal, "home"),
        "failed_to_score_away":  _safe(nogoal, "away"),
        "failed_to_score_total": _safe(nogoal, "total"),
        # Biggest results
        "biggest_win_home":  _safe(bigwin,  "home"),
        "biggest_win_away":  _safe(bigwin,  "away"),
        "biggest_loss_home": _safe(bigloss, "home"),
        "biggest_loss_away": _safe(bigloss, "away"),
        # Current streak
        "current_streak": (
            _safe(streak, "wins") and f"W{_safe(streak, 'wins')}"
            or _safe(streak, "draws") and f"D{_safe(streak, 'draws')}"
            or _safe(streak, "loses") and f"L{_safe(streak, 'loses')}"
        ),
        "raw_json":    json.dumps(raw),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Collection ─────────────────────────────────────────────────────────────────

def get_teams_in_league_season(conn, league_id, season):
    """
    Get distinct team IDs from the fixtures table for a given league/season.
    Must have run collect_fixtures.py first.
    """
    cur = conn.execute("""
        SELECT DISTINCT home_team_id, home_team FROM fixtures
        WHERE league_id = ? AND season = ?
    """, (league_id, season))
    return [(row[0], row[1]) for row in cur.fetchall()]


def collect_team_season_stats(conn, team_id, league_id, season, force=False):
    """
    Fetch and save stats for one team in one league/season.
    Returns True if new data was saved, False if skipped.
    """
    if not force:
        cur = conn.execute(
            "SELECT id FROM team_stats WHERE team_id=? AND league_id=? AND season=?",
            (team_id, league_id, season))
        if cur.fetchone():
            return False  # already collected

    raw_list = api_get("teams/statistics", {
        "team":   team_id,
        "league": league_id,
        "season": season,
    })

    if not raw_list:
        print(f"    ↳ No data returned for team {team_id}")
        return False

    row = parse_team_stats(raw_list[0] if isinstance(raw_list, list) else raw_list,
                           league_id, season)

    if not row["team_id"]:
        return False

    conn.execute("""
        INSERT OR REPLACE INTO team_stats (
            team_id, team_name, league_id, season,
            played_home, played_away, played_total,
            wins_home, wins_away, wins_total,
            draws_home, draws_away, draws_total,
            losses_home, losses_away, losses_total,
            goals_for_home, goals_for_away, goals_for_total,
            goals_against_home, goals_against_away, goals_against_total,
            goals_for_avg_home, goals_for_avg_away, goals_for_avg_total,
            goals_against_avg_home, goals_against_avg_away, goals_against_avg_total,
            shots_total, shots_on_target,
            possession_avg, pass_accuracy,
            yellow_cards, red_cards,
            clean_sheets_home, clean_sheets_away, clean_sheets_total,
            failed_to_score_home, failed_to_score_away, failed_to_score_total,
            biggest_win_home, biggest_win_away,
            biggest_loss_home, biggest_loss_away,
            current_streak, raw_json, collected_at
        ) VALUES (
            :team_id, :team_name, :league_id, :season,
            :played_home, :played_away, :played_total,
            :wins_home, :wins_away, :wins_total,
            :draws_home, :draws_away, :draws_total,
            :losses_home, :losses_away, :losses_total,
            :goals_for_home, :goals_for_away, :goals_for_total,
            :goals_against_home, :goals_against_away, :goals_against_total,
            :goals_for_avg_home, :goals_for_avg_away, :goals_for_avg_total,
            :goals_against_avg_home, :goals_against_avg_away, :goals_against_avg_total,
            :shots_total, :shots_on_target,
            :possession_avg, :pass_accuracy,
            :yellow_cards, :red_cards,
            :clean_sheets_home, :clean_sheets_away, :clean_sheets_total,
            :failed_to_score_home, :failed_to_score_away, :failed_to_score_total,
            :biggest_win_home, :biggest_win_away,
            :biggest_loss_home, :biggest_loss_away,
            :current_streak, :raw_json, :collected_at
        )
    """, row)
    conn.commit()
    return True


def export_parquet(conn, parquet_path):
    df = pd.read_sql("SELECT * FROM team_stats ORDER BY league_id, season, team_id", conn)
    df.to_parquet(parquet_path, index=False)
    print(f"\n  Exported {len(df):,} team-season records → {parquet_path}")
    return df


def print_summary(conn):
    print("\n  ── Team Stats Summary ────────────────────────────────────")
    rows = conn.execute("""
        SELECT ts.league_id, f.league_name, ts.season, COUNT(*) as teams
        FROM team_stats ts
        LEFT JOIN (
            SELECT DISTINCT league_id, league_name FROM fixtures
        ) f ON f.league_id = ts.league_id
        GROUP BY ts.league_id, ts.season
        ORDER BY ts.league_id, ts.season
    """).fetchall()
    for r in rows:
        print(f"  League {r[0]:4d}  {(r[1] or ''):30s}  {r[2]}  —  {r[3]} teams")
    total = conn.execute("SELECT COUNT(*) FROM team_stats").fetchone()[0]
    print(f"\n  Total: {total:,} team-season records")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect team statistics from API-Football")
    parser.add_argument("--update",    action="store_true",
                        help="Current season only (fits in ~100 daily requests)")
    parser.add_argument("--big5-only", action="store_true",
                        help="Only collect Big 5 leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1 + 2nd divs)")
    parser.add_argument("--league",    type=int, default=None,
                        help="Single league ID")
    parser.add_argument("--season",    type=int, default=None,
                        help="Single season (e.g. 2024)")
    parser.add_argument("--force",     action="store_true",
                        help="Re-fetch even if data already exists")
    parser.add_argument("--db",        default=None,
                        help="SQLite database path")
    parser.add_argument("--yes",       action="store_true",
                        help="Skip the >90 requests confirmation (use on Pro plan)")
    args = parser.parse_args()

    from config.settings import DB_PATH, PARQUET_DIR
    db_path      = args.db or str(DB_PATH)
    parquet_path = str(PARQUET_DIR / "team_stats.parquet")

    conn = get_db(db_path)

    # Determine which leagues and seasons
    if args.league:
        league_ids = [args.league]
    elif args.big5_only:
        league_ids = BIG5_IDS
    else:
        league_ids = [l["id"] for l in LEAGUES]

    if args.update:
        seasons = [CURRENT_SEASON]
    elif args.season:
        seasons = [args.season]
    else:
        seasons = SEASONS + [CURRENT_SEASON]

    # Build the work list: get team IDs from fixtures table
    print(f"\n── Collecting team statistics ───────────────────────────────")
    print(f"  Leagues:   {len(league_ids)}")
    print(f"  Seasons:   {seasons}")
    print(f"  ℹ️   Fixtures must already be collected (run collect_fixtures.py first)")
    print(f"  ⚠️  Free tier: 100 requests/day — each team = 1 request\n")

    # Count teams we need to collect
    work_list = []  # (team_id, team_name, league_id, season)
    for league_id in league_ids:
        for season in seasons:
            teams = get_teams_in_league_season(conn, league_id, season)
            if not teams:
                print(f"  ⚠️  No fixtures found for league {league_id} season {season} "
                      f"— run collect_fixtures.py first")
                continue
            for team_id, team_name in teams:
                if team_id:
                    work_list.append((team_id, team_name, league_id, season))

    # Deduplicate (same team_id + season can appear in multiple leagues — rare, keep all)
    seen = set()
    deduped = []
    for item in work_list:
        key = (item[0], item[2], item[3])  # team_id, league_id, season
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    print(f"  Teams to collect: {len(deduped)}")

    if len(deduped) == 0:
        print("\n  ❌ Nothing to collect — run collect_fixtures.py first.")
        conn.close()
        return

    if len(deduped) > 90 and not args.yes:
        print(f"\n  ⚠️  {len(deduped)} requests will exceed 90/day free tier safe limit.")
        print(f"     On Pro plan (7,500/day) this is fine — pass --yes to skip this check.")
        print(f"     Or --big5-only to limit scope.")
        user_in = input("  Continue anyway? [y/N] ").strip().lower()
        if user_in != 'y':
            print("  Aborted.")
            conn.close()
            return

    total_saved  = 0
    total_skipped = 0

    for i, (team_id, team_name, league_id, season) in enumerate(deduped, 1):
        league_name = LEAGUE_BY_ID.get(league_id, {}).get("name", f"League {league_id}")
        print(f"  [{i:4d}/{len(deduped)}] {team_name:35s} — {league_name} {season}")
        try:
            saved = collect_team_season_stats(conn, team_id, league_id, season, args.force)
            if saved:
                total_saved += 1
            else:
                total_skipped += 1
                print(f"    ↳ Skipped (already collected)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

    print(f"\n  ✅ Done — {total_saved} new records, {total_skipped} skipped")
    print_summary(conn)
    export_parquet(conn, parquet_path)
    conn.close()


if __name__ == "__main__":
    main()
