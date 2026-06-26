# ⚽ Football & World Cup Match Prediction

A production-style sports-prediction system covering **two model families**:

1. **Club pipeline** — match outcomes (1X2), Over/Under 2.5, BTTS and scorelines across **24 leagues** (Europe + Americas), built on ~44,000 historical fixtures.
2. **International pipeline** — a dedicated **national-team model** for the **2026 World Cup**, built on ~5,700 senior internationals.

Both feed a **value-bet engine** that compares model probabilities against live bookmaker odds and flags positive-expected-value bets — with the calibration discipline to know when *not* to bet.

> Built end-to-end: data engineering → feature engineering → modelling → backtesting → odds integration → reporting.

---

## 📊 Measured performance (held-out test sets)

### Club models — XGBoost (test season, ~8,900 matches)
| Market | Metric | Score |
|--------|--------|-------|
| Match result (1X2) | Accuracy | **53.2%** |
| Over/Under 2.5 | Accuracy / AUC | **61.6% / 0.667** |
| Both Teams To Score | Accuracy / AUC | **59.8% / 0.639** |

### World Cup model — national-team Elo + Dixon-Coles Poisson (554 held-out competitive internationals)
| Market | Metric | Score |
|--------|--------|-------|
| Match result (1X2) | Accuracy | **65.3%** |
| Match result (1X2) | Log-loss | **0.776** |
| Draw calibration | model vs actual | **22.6% vs 21.7%** (+1.0%) |

*1X2 accuracy in the 53–65% range with well-calibrated probabilities is competitive with the betting market — the higher international figure reflects the larger talent gaps between national teams.*

---

## 🧠 How it works

### Club pipeline
```
API-Football ─┐
Odds API ─────┼─► SQLite ─► feature engineering ─► XGBoost (1X2/OU/BTTS) ─┐
NewsAPI ──────┘     (Elo, form, H2H,            └► Dixon-Coles Poisson ───┼─► value-bet report
                     injuries, sentiment)          (scorelines)            ┘
```
- **Features (123):** Elo ratings, rolling form (last 5/10), head-to-head, season goal averages, rest days, derby flags, injury counts, news-sentiment scores (Twitter/XLM-RoBERTa on 2,900+ articles).
- **XGBoost** — separate classifiers for 1X2, O/U 2.5 and BTTS, each tuned with **50 Optuna trials**.
- **Dixon-Coles Poisson** — per-league attack/defence strengths, home advantage, low-score correction (ρ) and exponential time-decay weighting.

### International pipeline
- **Data:** World Cups, continental championships (Euro, Copa, AFCON, Asian Cup, Gold Cup), Nations Leagues, all six confederations' WC qualifiers, and 2,600+ friendlies — filtered to **senior men's national teams only**.
- **National-team Elo** — competition-weighted K-factor, goal-difference multiplier, neutral-venue aware.
- **Neutral-aware Dixon-Coles Poisson** — home advantage applied only for genuine host matches; World Cup games scored as neutral.
- **Model selection by time-based backtest** (`--tune`): trains on pre-cutoff matches, scores log-loss/accuracy on later competitive games, grid-searching the time-decay and Poisson/Elo blend.

---

## 💰 Value-bet detection — and knowing when not to bet

A bet is flagged when the model probability beats the **de-vigged** market consensus by **>4%**:
```
edge = model_probability − devigged_market_probability      (flag if edge > 4%)
EV   = model_probability × best_available_odds − 1
```
Odds are pulled live from **40+ bookmakers** (The Odds API) and de-vigged before comparison; the best price across books is used for EV.

**Calibration guardrails (this is the important part).** The system validates each market against actuals before trusting it for value:
- **1X2** is backtested and well-calibrated → value flags are surfaced.
- During the World Cup run, the model's **Over 2.5** sat ~10% below the market across the whole slate, while its historical bias was only −2.7%. That gap was diagnosed as **model conservatism, not edge**, so O/U value flags were **suppressed** rather than bet. A model that disagrees with the market in one direction *every single time* is biased, not sharp — and the pipeline is built to catch that.

---

## 🗂️ Project structure
```
football-prediction/
├── run_pipeline.py                 # orchestration: --init / --update / --train / --predict
├── config/
│   ├── settings.py                 # env-based API keys, paths, rate limits
│   └── leagues.py                  # 24 league IDs + season conventions
├── data_collection/
│   ├── collect_fixtures.py         # historical results (API-Football)
│   ├── collect_odds.py             # pre-match odds (The Odds API, 40+ books)
│   ├── collect_team_stats.py       # team season stats
│   ├── collect_injuries.py         # injury data
│   ├── collect_sentiment.py        # news sentiment (Twitter/XLM-RoBERTa)
│   └── collect_internationals.py   # senior national-team match history
├── features/
│   └── build_features.py           # Elo, form, H2H, injuries, sentiment → 123 features
├── models/
│   ├── xgboost_model.py            # club 1X2 / OU2.5 / BTTS
│   ├── poisson_model.py            # club Dixon-Coles scorelines
│   └── national_team_model.py      # World Cup Elo + Dixon-Coles (+ --tune backtest)
└── predictions/
    ├── generate_report.py          # club value-bet report
    └── wc_report.py                # World Cup report + value bets
```
*Data, trained models and API keys are git-ignored — collect/train locally to reproduce.*

---

## 🚀 Quick start
```bash
pip install -r requirements.txt
cp .env.example .env          # add your API keys (API-Football, Odds API, NewsAPI)
source .env

# Club pipeline
python3 run_pipeline.py --init                # one-time historical load
python3 run_pipeline.py --train               # rebuild features + train XGBoost & Poisson
python3 run_pipeline.py --update --predict    # refresh data + value-bet report

# World Cup pipeline
python3 data_collection/collect_internationals.py   # national-team history
python3 models/national_team_model.py --tune        # backtest to choose hyperparameters
python3 models/national_team_model.py --train        # fit Elo + Dixon-Coles
python3 predictions/wc_report.py                     # predictions + value bets
```

---

## 🛠️ Tech stack
**Python** · scikit-learn · **XGBoost** · SciPy (custom Dixon-Coles MLE) · pandas / NumPy · **Optuna** · HuggingFace Transformers (sentiment) · SQLite · Parquet · The Odds API · API-Football

---

## 📈 Data scale
- **44,000+** club fixtures across 24 leagues, 5 seasons
- **5,700** senior international matches (2018–2026)
- **100,000+** injury records · **2,900+** sentiment-scored news articles
- **40+ bookmakers** for live odds

---

## 🗺️ Roadmap
- AWS serving: S3 + Lambda/EventBridge scheduled predictions, SNS email alerts
- Closing-line value tracking to measure real betting edge over time
- Lineup/rotation signal for tournament dead-rubbers
- Continuous backtest CI on each retrain

---

*Personal project demonstrating end-to-end ML engineering: data pipelines, feature engineering, classical + ML models, rigorous backtesting, and honest probability calibration.*
