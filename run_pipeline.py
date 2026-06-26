#!/usr/bin/env python3
"""
run_pipeline.py — Master orchestration script for the football prediction pipeline.

Runs the full pipeline in order:
  1. collect_fixtures    — pull historical results from API-Football
  2. collect_odds        — pull pre-match odds from the Odds API
  3. collect_team_stats  — pull team season stats (goals avg, possession, etc.)
  4. build_features      — compute Elo, form, H2H, context features
  5. train XGBoost       — train 1X2 / OU2.5 / BTTS classifiers
  6. train Poisson       — fit Dixon-Coles scoreline model
  7. generate report     — produce predictions email / summary

Usage:
  source .env

  # Initial data load (run once over 2 days due to API limits):
  python3 run_pipeline.py --init

  # Daily update (runs in <5 minutes, uses ~50 API requests):
  python3 run_pipeline.py --update

  # Train models on collected data:
  python3 run_pipeline.py --train

  # Generate predictions for upcoming matches:
  python3 run_pipeline.py --predict

  # Full pipeline (update data + predict):
  python3 run_pipeline.py --update --predict
"""

import os, sys, argparse, subprocess
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import FOOTBALL_API_KEY, ODDS_API_KEY


def run_step(name, cmd, required=True):
    """Run a pipeline step, exit on failure if required."""
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
    if result.returncode != 0 and required:
        print(f"\n❌ Step '{name}' failed — stopping pipeline")
        sys.exit(result.returncode)
    return result.returncode == 0


def check_keys():
    """Verify API keys are set."""
    missing = []
    if not FOOTBALL_API_KEY:
        missing.append("FOOTBALL_API_KEY")
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY")
    if missing:
        print(f"\n❌ Missing API keys: {', '.join(missing)}")
        print(f"   Run: source .env")
        sys.exit(1)
    print(f"  ✅ API keys loaded")
    print(f"  Football API: {FOOTBALL_API_KEY[:8]}...")
    print(f"  Odds API:     {ODDS_API_KEY[:8]}...")


def main():
    parser = argparse.ArgumentParser(description="Football prediction pipeline")
    parser.add_argument('--init',    action='store_true',
                        help='Initial data load: all 24 leagues, 5 seasons (needs 2 days)')
    parser.add_argument('--update',  action='store_true',
                        help='Daily update: current season + latest odds')
    parser.add_argument('--train',   action='store_true',
                        help='Train/retrain all models')
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions for upcoming matches')
    parser.add_argument('--big5',    action='store_true',
                        help='Limit collection to Big 5 leagues only')
    args = parser.parse_args()

    if not any([args.init, args.update, args.train, args.predict]):
        parser.print_help()
        sys.exit(0)

    start = datetime.now()
    print(f"\n{'═'*60}")
    print(f"  FOOTBALL PREDICTION PIPELINE")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*60}")

    check_keys()

    big5_flag = "--big5-only" if args.big5 else ""

    # ── Initial load ───────────────────────────────────────────────────────────
    if args.init:
        print("\n⚠️  INITIAL LOAD — this requires ~170 API requests across 2 days")
        print("   The script will warn you before exceeding daily limits.")
        print("   Press Ctrl+C to stop and resume tomorrow with --update\n")

        # Day 1: fixtures for 5 past seasons (120 requests)
        run_step("Collect fixtures (5 seasons)",
                 f"python3 data_collection/collect_fixtures.py")

        # Team stats (up to 100/day — run with big5 first)
        run_step("Collect team stats (Big 5 first)",
                 f"python3 data_collection/collect_team_stats.py --big5-only")

        # Odds: collect all upcoming
        run_step("Collect odds (all leagues)",
                 f"python3 data_collection/collect_odds.py")

    # ── Daily update ───────────────────────────────────────────────────────────
    if args.update:
        # Update current season results (24 requests)
        run_step("Update fixtures (current season)",
                 f"python3 data_collection/collect_fixtures.py --update")

        # Update odds for upcoming matches
        run_step("Update odds",
                 f"python3 data_collection/collect_odds.py")

        # Update team stats for current season (optional, ~20-100 requests)
        run_step("Update team stats (current season)",
                 f"python3 data_collection/collect_team_stats.py --update {big5_flag}",
                 required=False)

    # ── Build features ─────────────────────────────────────────────────────────
    if args.train:
        run_step("Build features (training set)",
                 f"python3 features/build_features.py")

    if args.predict:
        run_step("Build features (training + upcoming)",
                 f"python3 features/build_features.py --upcoming")

    # ── Train models ───────────────────────────────────────────────────────────
    if args.train:
        run_step("Train XGBoost model",
                 f"python3 models/xgboost_model.py --train")

        run_step("Fit Poisson model",
                 f"python3 models/poisson_model.py --train")

    # ── Generate predictions ───────────────────────────────────────────────────
    if args.predict:
        run_step("XGBoost predictions",
                 f"python3 models/xgboost_model.py --predict data/parquet/features.parquet",
                 required=False)

        run_step("Poisson scoreline predictions",
                 f"python3 models/poisson_model.py --predict-upcoming",
                 required=False)

        run_step("Generate predictions report",
                 f"python3 predictions/generate_report.py",
                 required=False)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'═'*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.0f}s")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
