"""
Settings — reads API keys from environment variables.
Set these before running any collection script:

  export FOOTBALL_API_KEY="your_api_football_key"
  export ODDS_API_KEY="your_odds_api_key"
  export NEWS_API_KEY="your_newsapi_key"
"""

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
ODDS_API_KEY     = os.environ.get("ODDS_API_KEY", "")
NEWS_API_KEY     = os.environ.get("NEWS_API_KEY", "")

# ── API Endpoints ─────────────────────────────────────────────────────────────
FOOTBALL_API_BASE = "https://v3.football.api-sports.io"
ODDS_API_BASE     = "https://api.the-odds-api.com/v4"
NEWS_API_BASE     = "https://newsapi.org/v2"

# ── Rate limits ───────────────────────────────────────────────────────────────
# API-Football free:  100 req/day,   10 req/minute  → 7s delay
# API-Football Pro:   7,500 req/day, 100 req/minute → 1s delay
# Odds API free:      500 req/month
_pro_plan = os.environ.get("FOOTBALL_PRO_PLAN", "0") == "1"
FOOTBALL_API_DELAY_SECONDS = 1.0 if _pro_plan else 7.0
ODDS_API_DELAY_SECONDS     = 2

# ── Local data storage ────────────────────────────────────────────────────────
import pathlib
BASE_DIR     = pathlib.Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data"
DB_PATH      = DATA_DIR / "football.db"      # SQLite for local dev
PARQUET_DIR  = DATA_DIR / "parquet"          # Parquet files for model training

DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

# ── AWS (for production deployment later) ────────────────────────────────────
AWS_REGION  = "eu-west-2"
S3_BUCKET   = "bitcoin-prediction-option4-production-654654488711"
S3_PREFIX   = "football"
