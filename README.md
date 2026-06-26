# Football Prediction Pipeline

Predict match outcomes (Win/Draw/Loss), scorelines, Over/Under 2.5 goals, and BTTS across **24 leagues** — Big 5 Europe, Extra European leagues, and Americas. Primary use: **value bet detection** where model probability exceeds bookmaker implied probability by >4%.

---

## Quick Start

```bash
cd /Users/terrysmac/project/football-prediction

# 1. Set API keys
source .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initial data load (first time — runs across 2 days due to 100 req/day limit)
python3 run_pipeline.py --init

# 4. Train models (after data is collected)
python3 run_pipeline.py --train

# 5. Daily: update data + generate predictions
python3 run_pipeline.py --update --predict
```

---

## Project Structure

```
football-prediction/
├── .env                          # API keys (NOT committed to git)
├── run_pipeline.py               # Master orchestration script
├── requirements.txt
│
├── config/
│   ├── settings.py               # API keys, paths, rate limits
│   └── leagues.py                # 24 leagues with API-Football IDs
│
├── data_collection/
│   ├── collect_fixtures.py       # Historical match results (API-Football)
│   ├── collect_team_stats.py     # Team season stats (goals, possession, etc.)
│   └── collect_odds.py           # Pre-match odds (Odds API)
│
├── features/
│   └── build_features.py         # Elo, form, H2H, context — all features
│
├── models/
│   ├── xgboost_model.py          # XGBoost 1X2 / OU2.5 / BTTS ensemble
│   ├── poisson_model.py          # Dixon-Coles Poisson scoreline model
│   └── saved/                    # Trained model artifacts
│
├── predictions/
│   ├── generate_report.py        # Merge models, flag value bets, email report
│   └── reports/                  # Daily prediction files (text + JSON)
│
└── data/
    ├── football.db               # SQLite: fixtures, odds, team_stats
    └── parquet/                  # Processed data for model training
```

---

## Data Sources

| Source | What | Free Tier |
|--------|------|-----------|
| API-Football | Fixtures, results, team stats | 100 req/day |
| Odds API | Pre-match odds (40+ bookmakers) | 500 req/month |
| NewsAPI | Injuries, transfers, team news | Existing key |

---

## API Request Budget (Free Tier)

### Initial load (one-time, spread over ~3 days)
| Step | Requests |
|------|----------|
| Fixtures: 24 leagues × 6 seasons | 144 |
| Team stats: ~400 teams × 6 seasons | 2,400 (spread over months) |
| Odds: 19 sport keys | 19 |

### Daily update
| Step | Requests |
|------|----------|
| Fixtures (current season) | 24 |
| Odds (all upcoming) | 19 |
| Team stats (optional) | 0–24 |
| **Total** | **~43/day** ✅ |

---

## Models

### 1. XGBoost Ensemble
- **Predicts**: Home Win / Draw / Away Win probabilities + Over/Under 2.5 + BTTS
- **Features**: Rolling form (last 5, 10 matches), Elo ratings, H2H record, season averages, rest days, derby flag
- **Training**: Seasons 2020–2023 → validate 2024 → predict 2025
- **Tuning**: 50 Optuna trials per model

### 2. Dixon-Coles Poisson
- **Predicts**: Full scoreline distribution (P(0-0), P(1-0), ... P(6-6))
- **Outputs**: Most likely scoreline, expected goals, aggregated H/D/A probs
- **Features**: Attack strength + defence weakness per team, home advantage, time-decay weighting
- **Classic method**: Used by professional betting syndicates since 1997

### Ensemble
Both models' H/D/A probabilities are averaged for the final prediction.

---

## Value Bet Detection

A **value bet** is flagged when:
```
model_probability > bookmaker_implied_probability + 4%
```

Bookmaker implied probability = 1 / decimal_odds (e.g. odds 2.50 → implied = 40%)

If the model gives 46% chance of a home win but the bookmaker implies only 40%, that's a 6% edge = value bet.

**Second divisions** tend to have the biggest inefficiencies — bookmakers focus attention on the Premier League.

---

## 24 Leagues

### Big 5 Europe
Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2

### Extra European
Eredivisie, Eerste Divisie, Primeira Liga, Liga Portugal 2, Süper Lig, TFF First League, Pro League, Austrian Bundesliga, Austrian 2. Liga

### Americas
MLS, Liga MX, Brazilian Série A, Argentine Primera División

---

## AWS Deployment (future)

Mirrors the Bitcoin prediction pipeline:
- **S3**: `bitcoin-prediction-option4-production-654654488711/football/`
- **DynamoDB**: `football-predictions-production`
- **Lambda**: trigger 2h before each match day (EventBridge)
- **SNS**: email predictions to oluwaseyifamuyide@gmail.com
- **SageMaker**: endpoints for XGBoost and Poisson models
