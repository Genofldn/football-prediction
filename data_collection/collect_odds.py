#!/usr/bin/env python3
"""
collect_odds.py — Pull pre-match odds from the Odds API.

Odds API free tier: 500 requests/month (~16/day).
Strategy:
  - Fetch upcoming fixtures odds per league as needed (not all at once)
  - Focus on 1X2 market (match result) + Over/Under 2.5 + BTTS
  - Run before each match day to get latest odds

Odds are the STRONGEST single signal — bookmaker implied probabilities
represent the market's best estimate. Our model aims to beat them by 4%+.

Usage:
  export ODDS_API_KEY="your_key_here"
  python3 collect_odds.py                  # fetch odds for all upcoming matches
  python3 collect_odds.py --sport soccer_epl  # single league
  python3 collect_odds.py --list-sports    # show available sport keys
"""

import os, sys, time, json, sqlite3, argparse, requests
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import ODDS_API_KEY, ODDS_API_BASE, ODDS_API_DELAY_SECONDS

import pandas as pd


# ── Odds API sport keys for our 24 leagues ─────────────────────────────────────
# Map our league IDs to Odds API sport keys
LEAGUE_TO_ODDS_SPORT = {
    39:  "soccer_epl",               # Premier League
    40:  "soccer_efl_champ",         # Championship
    140: "soccer_spain_la_liga",     # La Liga
    141: "soccer_spain_segunda_division",  # La Liga 2
    78:  "soccer_germany_bundesliga",     # Bundesliga
    79:  "soccer_germany_bundesliga2",    # 2. Bundesliga
    135: "soccer_italy_serie_a",     # Serie A
    136: "soccer_italy_serie_b",     # Serie B
    61:  "soccer_france_ligue_one",  # Ligue 1
    62:  "soccer_france_ligue_two",  # Ligue 2
    88:  "soccer_netherlands_eredivisie",  # Eredivisie
    94:  "soccer_portugal_primeira_liga",  # Primeira Liga
    203: "soccer_turkey_super_league",    # Süper Lig
    144: "soccer_belgium_first_div",      # Pro League
    218: "soccer_austria_bundesliga",     # Austrian Bundesliga
    253: "soccer_usa_mls",           # MLS
    262: "soccer_mexico_ligamx",     # Liga MX
    71:  "soccer_brazil_campeonato", # Brazilian Série A
    128: "soccer_argentina_primera_division",  # Argentine Primera
}

# Markets to collect
MARKETS = ["h2h", "totals", "btts"]  # h2h=1X2, totals=over/under, btts=both teams score

# Preferred bookmakers (prioritise for odds comparison)
PREFERRED_BOOKMAKERS = [
    "bet365", "betfair", "unibet", "williamhill",
    "paddypower", "ladbrokes", "coral"
]


def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sport_key       TEXT,
            event_id        TEXT,
            commence_time   TEXT,
            home_team       TEXT,
            away_team       TEXT,
            market          TEXT,
            bookmaker       TEXT,
            outcome_name    TEXT,
            outcome_price   REAL,
            implied_prob    REAL,
            collected_at    TEXT,
            UNIQUE(event_id, market, bookmaker, outcome_name)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_event ON odds(event_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_sport ON odds(sport_key)")
    conn.commit()
    return conn


def api_get_odds(sport_key, markets):
    """Fetch upcoming odds for a sport from the Odds API."""
    if not ODDS_API_KEY:
        raise EnvironmentError(
            "ODDS_API_KEY not set. Run: export ODDS_API_KEY='your_key'")

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": "uk,eu",
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)

    # Show remaining quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"    [Odds API] {sport_key} → {remaining} requests remaining "
          f"({used} used this month)", flush=True)

    if resp.status_code == 401:
        raise RuntimeError("Invalid ODDS_API_KEY")
    if resp.status_code == 422:
        print(f"    ⚠️  Sport '{sport_key}' not available on free tier — skipping")
        return []
    resp.raise_for_status()

    time.sleep(ODDS_API_DELAY_SECONDS)
    return resp.json()


def parse_and_save_odds(conn, events, sport_key):
    """Parse Odds API events and store each bookmaker line."""
    rows = []
    now  = datetime.now(timezone.utc).isoformat()

    for event in events:
        event_id     = event.get("id")
        commence     = event.get("commence_time", "")[:10]
        home_team    = event.get("home_team")
        away_team    = event.get("away_team")

        for bookmaker in event.get("bookmakers", []):
            bk_key = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    implied_prob = round(1.0 / price, 4) if price and price > 0 else None
                    rows.append({
                        "sport_key":     sport_key,
                        "event_id":      event_id,
                        "commence_time": commence,
                        "home_team":     home_team,
                        "away_team":     away_team,
                        "market":        market_key,
                        "bookmaker":     bk_key,
                        "outcome_name":  outcome.get("name"),
                        "outcome_price": price,
                        "implied_prob":  implied_prob,
                        "collected_at":  now,
                    })

    if rows:
        conn.executemany("""
            INSERT OR REPLACE INTO odds (
                sport_key, event_id, commence_time, home_team, away_team,
                market, bookmaker, outcome_name, outcome_price,
                implied_prob, collected_at
            ) VALUES (
                :sport_key, :event_id, :commence_time, :home_team, :away_team,
                :market, :bookmaker, :outcome_name, :outcome_price,
                :implied_prob, :collected_at
            )
        """, rows)
        conn.commit()
        print(f"    ↳ Saved {len(rows)} odds lines for {len(events)} events")

    return len(rows)


def get_consensus_odds(conn, event_id=None, sport_key=None):
    """
    Calculate consensus (average) implied probability across bookmakers.
    Returns a DataFrame with one row per event/market/outcome.
    Most useful for model features — use the market consensus, not one bookmaker.
    """
    query = """
        SELECT
            event_id, sport_key, commence_time, home_team, away_team,
            market, outcome_name,
            AVG(implied_prob)  AS avg_implied_prob,
            MAX(outcome_price) AS best_price,
            COUNT(DISTINCT bookmaker) AS bookmaker_count
        FROM odds
        WHERE 1=1
    """
    params = []
    if event_id:
        query += " AND event_id = ?"
        params.append(event_id)
    if sport_key:
        query += " AND sport_key = ?"
        params.append(sport_key)

    query += " GROUP BY event_id, market, outcome_name ORDER BY commence_time, event_id"
    return pd.read_sql(query, conn, params=params)


def main():
    parser = argparse.ArgumentParser(description="Collect pre-match odds from the Odds API")
    parser.add_argument("--sport",       default=None,
                        help="Collect only this Odds API sport key (e.g. soccer_epl)")
    parser.add_argument("--list-sports", action="store_true",
                        help="List all configured sport keys and exit")
    parser.add_argument("--db",          default=None)
    args = parser.parse_args()

    if args.list_sports:
        print("\nConfigured sport keys:")
        for lid, sk in LEAGUE_TO_ODDS_SPORT.items():
            print(f"  League {lid:4d}  →  {sk}")
        sys.exit(0)

    from config.settings import DB_PATH
    conn = get_db(args.db or str(DB_PATH))

    if args.sport:
        sport_keys = [args.sport]
    else:
        # Only collect sports for active/upcoming fixtures
        sport_keys = list(set(LEAGUE_TO_ODDS_SPORT.values()))

    print(f"\n── Collecting odds ──────────────────────────────────────────")
    print(f"  Sport keys: {len(sport_keys)}")
    print(f"  Markets:    {MARKETS}")
    print(f"  ⚠️  Free tier: 500 requests/month. Each sport key = 1 request.\n")

    total_lines = 0
    for sport_key in sport_keys:
        print(f"  {sport_key}")
        try:
            events = api_get_odds(sport_key, MARKETS)
            if events:
                total_lines += parse_and_save_odds(conn, events, sport_key)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

    print(f"\n  ✅ Done — {total_lines:,} odds lines saved")
    conn.close()


if __name__ == "__main__":
    main()
