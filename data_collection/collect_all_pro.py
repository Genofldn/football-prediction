#!/usr/bin/env python3
"""
collect_all_pro.py — Full data collection using the Pro plan (100 req/min).

Run this ONCE after subscribing to the Pro plan.
It collects everything in one session:
  1. Fixtures for 2021 + 2025 seasons (free tier already has 2022–2024)
  2. Team stats for all teams × all seasons (≈400 teams × 5 seasons = ~2000 requests)

At 100 req/min this takes about 25-30 minutes total.
Rate limit: we use 1.5s delay between requests to stay safely under 100/min.

Usage:
  source .env
  python3 data_collection/collect_all_pro.py
  python3 data_collection/collect_all_pro.py --fixtures-only
  python3 data_collection/collect_all_pro.py --stats-only
"""

import os, sys, time, sqlite3, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings     import DB_PATH, PARQUET_DIR, FOOTBALL_API_DELAY_SECONDS
from config.leagues      import LEAGUES, LEAGUE_BY_ID

# On pro plan: reduce delay from 7s → 1.5s (100 req/min limit)
PRO_DELAY = 1.5

# Seasons to collect (2021 and 2025 are not on free tier)
NEW_SEASONS   = [2021, 2025]
ALL_SEASONS   = [2021, 2022, 2023, 2024, 2025]


def set_pro_delay():
    """Monkey-patch the delay for pro plan speed."""
    import config.settings as cs
    cs.FOOTBALL_API_DELAY_SECONDS = PRO_DELAY
    print(f"  ⚡ Pro plan mode: request delay reduced to {PRO_DELAY}s")


def main():
    parser = argparse.ArgumentParser(description="Full data collection on Pro plan")
    parser.add_argument('--fixtures-only', action='store_true')
    parser.add_argument('--stats-only',    action='store_true')
    parser.add_argument('--big5-only',     action='store_true',
                        help='Limit team stats to Big 5 leagues (faster)')
    parser.add_argument('--db',            default=None)
    args = parser.parse_args()

    db_path = args.db or str(DB_PATH)
    set_pro_delay()

    do_fixtures = not args.stats_only
    do_stats    = not args.fixtures_only

    if do_fixtures:
        print(f"\n{'═'*60}")
        print(f"  STEP 1 — Collect missing seasons (2021 + 2025)")
        print(f"{'═'*60}")
        for season in NEW_SEASONS:
            cmd = (f"python3 data_collection/collect_fixtures.py "
                   f"--season {season}")
            print(f"\n  > {cmd}")
            os.system(f"cd {os.path.join(os.path.dirname(__file__), '..')} && {cmd}")

    if do_stats:
        print(f"\n{'═'*60}")
        print(f"  STEP 2 — Collect team statistics (all teams × all seasons)")
        print(f"{'═'*60}")
        print(f"  ⚠️  This is ~2,000 requests at 1.5s delay = ~50 minutes")
        print(f"  Team stats power the season average features in the model.\n")

        big5_flag = "--big5-only" if args.big5_only else ""
        for season in ALL_SEASONS:
            cmd = (f"python3 data_collection/collect_team_stats.py "
                   f"--season {season} {big5_flag}")
            print(f"\n  > {cmd}")
            os.system(f"cd {os.path.join(os.path.dirname(__file__), '..')} && {cmd}")

    print(f"\n{'═'*60}")
    print(f"  ✅ Pro collection complete!")
    print(f"  Next: python3 run_pipeline.py --train")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
