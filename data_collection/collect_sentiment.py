#!/usr/bin/env python3
"""
collect_sentiment.py — Pull team/match news and score with Twitter-RoBERTa.

Sentiment engine:
  PRIMARY:   cardiffnlp/twitter-roberta-base-sentiment-latest
             Trained on 124M tweets — football breaking news is written in
             exactly this style. Understands context VADER misses:
               "Salah doubtful for weekend clash" → -0.87 (VADER: 0.0)
               "Haaland returns to training"      → +0.81 (VADER: +0.1)

  CLASSIFIER: cross-encoder/nli-deberta-v3-base (zero-shot)
              Labels each article without any training:
              ["key player injury", "suspension", "manager sacked",
               "player returning", "team crisis", "transfer rumour"]

  MULTILINGUAL: cardiffnlp/twitter-xlm-roberta-base-sentiment
                Handles German, French, Spanish, Portuguese, Turkish —
                so Bundesliga/Ligue 2/Süper Lig teams get real coverage
                from their local-language news feeds.

Runs on Apple M2 MPS (GPU) — ~100 articles/sec, much faster than CPU.
Models download once (~500MB each) then cached locally.

Usage:
  source .env
  python3 collect_sentiment.py               # all Big 5 teams, last 7 days
  python3 collect_sentiment.py --days 30     # last 30 days (initial load)
  python3 collect_sentiment.py --all-leagues # all 24 leagues (uses multilingual)
  python3 collect_sentiment.py --team "Arsenal"
"""

import os, sys, time, json, sqlite3, argparse, requests, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import NEWS_API_KEY, NEWS_API_BASE, DB_PATH, PARQUET_DIR
from config.leagues  import LEAGUES, LEAGUE_BY_ID, BIG5_IDS

import pandas as pd
import torch

# ── Device setup (M2 MPS > CUDA > CPU) ────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"  Sentiment engine: twitter-roberta on {DEVICE.upper()}", flush=True)

# ── Load models (cached after first download) ──────────────────────────────────

_sentiment_pipe    = None   # English: twitter-roberta
_sentiment_pipe_ml = None   # Multilingual: xlm-roberta
_classifier_pipe   = None   # Zero-shot NLI

def get_sentiment_pipe(multilingual=False):
    global _sentiment_pipe, _sentiment_pipe_ml
    from transformers import pipeline

    if multilingual:
        if _sentiment_pipe_ml is None:
            print("  Loading multilingual model (twitter-xlm-roberta)...", flush=True)
            _sentiment_pipe_ml = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=DEVICE,
                truncation=True,
                max_length=512,
                top_k=None,       # return all class scores
            )
        return _sentiment_pipe_ml
    else:
        if _sentiment_pipe is None:
            print("  Loading english model (twitter-roberta-base-sentiment-latest)...", flush=True)
            _sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=DEVICE,
                truncation=True,
                max_length=512,
                top_k=None,       # return all class scores
            )
        return _sentiment_pipe


def get_classifier_pipe():
    global _classifier_pipe
    if _classifier_pipe is None:
        from transformers import pipeline
        print("  Loading zero-shot classifier (nli-deberta-v3-base)...", flush=True)
        _classifier_pipe = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=DEVICE,
        )
    return _classifier_pipe


# ── Impact labels for zero-shot classification ─────────────────────────────────

IMPACT_LABELS = [
    "key player injury or fitness doubt",
    "player suspension or ban",
    "manager sacked or resigned",
    "player returning from injury",
    "team crisis or dressing room unrest",
    "transfer news or rumour",
    "team winning run or positive form",
    "neutral football news",
]

# Map label back to column flags
LABEL_TO_FLAG = {
    "key player injury or fitness doubt":   "has_injury",
    "player suspension or ban":             "has_suspension",
    "manager sacked or resigned":           "has_manager_change",
    "player returning from injury":         "has_positive",
    "team crisis or dressing room unrest":  "has_crisis",
    "transfer news or rumour":              "has_transfer",
    "team winning run or positive form":    "has_positive",
    "neutral football news":               None,
}


def roberta_score(texts, multilingual=False, batch_size=32):
    """
    Score a list of texts with Twitter-RoBERTa.
    Returns list of compound scores: -1.0 (very negative) to +1.0 (very positive).
    """
    if not texts:
        return []

    pipe = get_sentiment_pipe(multilingual)

    # Batch inference — much faster than one-by-one
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            outputs = pipe(batch)
        except Exception:
            # Fallback: process one at a time if batch fails
            outputs = []
            for t in batch:
                try:
                    outputs.append(pipe([t])[0])
                except Exception:
                    outputs.append([])

        for result in outputs:
            if not result:
                all_results.append(0.0)
                continue

            # result is a list of {label, score} for all classes
            scores = {r['label'].lower(): r['score'] for r in result}

            # Compound score: positive - negative (weighted by confidence)
            pos = scores.get('positive', scores.get('pos', 0.0))
            neg = scores.get('negative', scores.get('neg', 0.0))
            neu = scores.get('neutral',  scores.get('neu', 0.0))

            # Compound: scale to -1..+1
            compound = pos - neg
            all_results.append(round(compound, 4))

    return all_results


def classify_impact(texts, batch_size=16):
    """
    Zero-shot classify texts into football impact categories.
    Returns list of dicts with impact flags.
    """
    if not texts:
        return [_empty_flags() for _ in texts]

    classifier = get_classifier_pipe()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            outputs = classifier(batch, IMPACT_LABELS, multi_label=True)
            if not isinstance(outputs, list):
                outputs = [outputs]

            for output in outputs:
                flags = _empty_flags()
                # Flag labels where confidence > 0.5
                for label, score in zip(output['labels'], output['scores']):
                    if score > 0.50:
                        col = LABEL_TO_FLAG.get(label)
                        if col:
                            flags[col] = 1
                results.append(flags)

        except Exception:
            results.extend([_empty_flags() for _ in batch])

    return results


def _empty_flags():
    return {
        'has_injury':        0,
        'has_suspension':    0,
        'has_manager_change': 0,
        'has_positive':      0,
        'has_crisis':        0,
        'has_transfer':      0,
    }


# ── Language detection (simple heuristic) ─────────────────────────────────────

# Leagues where local-language news is more available than English
MULTILINGUAL_LEAGUE_IDS = {
    78, 79,    # Germany
    61, 62,    # France
    88, 89,    # Netherlands
    203, 204,  # Turkey
    218, 219,  # Austria
    71,        # Brazil
    128,       # Argentina
    262,       # Mexico
}


def is_multilingual_league(league_id):
    return league_id in MULTILINGUAL_LEAGUE_IDS


# ── Database ───────────────────────────────────────────────────────────────────

def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id         INTEGER,
            team_name       TEXT,
            league_id       INTEGER,
            article_url     TEXT UNIQUE,
            headline        TEXT,
            description     TEXT,
            published_at    TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,   -- 'positive' / 'negative' / 'neutral'
            -- Impact flags (zero-shot NLI)
            has_injury          INTEGER DEFAULT 0,
            has_suspension      INTEGER DEFAULT 0,
            has_manager_change  INTEGER DEFAULT 0,
            has_positive        INTEGER DEFAULT 0,
            has_crisis          INTEGER DEFAULT 0,
            has_transfer        INTEGER DEFAULT 0,
            -- Source
            source_name     TEXT,
            sentiment_engine TEXT,  -- 'twitter-roberta' / 'xlm-roberta'
            collected_at    TEXT
        )
    """)
    # Add new columns if upgrading from old schema
    for col_def in [
        "ALTER TABLE news_sentiment ADD COLUMN has_transfer INTEGER DEFAULT 0",
        "ALTER TABLE news_sentiment ADD COLUMN sentiment_label TEXT",
        "ALTER TABLE news_sentiment ADD COLUMN sentiment_engine TEXT",
    ]:
        try:
            conn.execute(col_def)
        except Exception:
            pass  # Column already exists

    conn.execute("CREATE INDEX IF NOT EXISTS idx_ns_team   ON news_sentiment(team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ns_date   ON news_sentiment(published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ns_league ON news_sentiment(league_id)")
    conn.commit()
    return conn


# ── NewsAPI ────────────────────────────────────────────────────────────────────

NEWS_API_DELAY = 0.5  # seconds between successful requests (was 1.2)


def fetch_team_news(team_name, from_date, to_date, language='en'):
    """Fetch recent news for a team from NewsAPI."""
    if not NEWS_API_KEY:
        raise EnvironmentError("NEWS_API_KEY not set")

    params = {
        "q":        f'"{team_name}" football',
        "from":     from_date.strftime("%Y-%m-%d"),
        "to":       to_date.strftime("%Y-%m-%d"),
        "language": language,
        "sortBy":   "publishedAt",
        "pageSize": 50,
        "apiKey":   NEWS_API_KEY,
    }

    for attempt in range(4):  # retry up to 3 times on 429
        resp = requests.get(f"{NEWS_API_BASE}/everything", params=params, timeout=20)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
            print(f"\n    ⚠️  NewsAPI rate limit — sleeping {wait}s (attempt {attempt+1}/4)", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            return []
        time.sleep(NEWS_API_DELAY)
        return resp.json().get("articles", [])

    return []  # all retries exhausted


# ── Process and save ───────────────────────────────────────────────────────────

def process_and_save(conn, articles, team_id, team_name, league_id,
                     use_classifier=True):
    """Score headlines with RoBERTa and save to DB."""
    if not articles:
        return 0

    multilingual = is_multilingual_league(league_id or 0)
    engine_name  = "xlm-roberta" if multilingual else "twitter-roberta"

    # Build text list for batch inference
    texts = []
    for a in articles:
        headline = a.get("title", "") or ""
        desc     = a.get("description", "") or ""
        # RoBERTa works best on short texts — headline is sufficient
        texts.append(headline[:280])  # Tweet-length

    # Batch sentiment scoring
    scores = roberta_score(texts, multilingual=multilingual)

    # Batch impact classification (optional — slower)
    if use_classifier:
        flags_list = classify_impact(texts)
    else:
        flags_list = [_empty_flags() for _ in texts]

    # Determine label from score
    def score_to_label(s):
        if s > 0.15:  return 'positive'
        if s < -0.15: return 'negative'
        return 'neutral'

    now = datetime.utcnow().isoformat()
    rows = []
    for article, score, flags in zip(articles, scores, flags_list):
        rows.append({
            "team_id":         team_id,
            "team_name":       team_name,
            "league_id":       league_id,
            "article_url":     article.get("url", ""),
            "headline":        (article.get("title", "") or "")[:500],
            "description":     (article.get("description", "") or "")[:1000],
            "published_at":    (article.get("publishedAt", "") or "")[:10],
            "sentiment_score": score,
            "sentiment_label": score_to_label(score),
            "source_name":     article.get("source", {}).get("name", ""),
            "sentiment_engine": engine_name,
            "collected_at":    now,
            **flags,
        })

    if rows:
        conn.executemany("""
            INSERT OR IGNORE INTO news_sentiment (
                team_id, team_name, league_id, article_url,
                headline, description, published_at,
                sentiment_score, sentiment_label,
                has_injury, has_suspension, has_manager_change,
                has_positive, has_crisis, has_transfer,
                source_name, sentiment_engine, collected_at
            ) VALUES (
                :team_id, :team_name, :league_id, :article_url,
                :headline, :description, :published_at,
                :sentiment_score, :sentiment_label,
                :has_injury, :has_suspension, :has_manager_change,
                :has_positive, :has_crisis, :has_transfer,
                :source_name, :sentiment_engine, :collected_at
            )
        """, rows)
        conn.commit()

    return len(rows)


# ── Feature extraction (used by build_features.py) ────────────────────────────

def get_sentiment_features(conn, team_id, match_date, days_back=7):
    """
    Aggregate RoBERTa sentiment for a team in the N days before a match.
    Used during feature engineering.
    """
    from_date = (pd.Timestamp(match_date) - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    cur = conn.execute("""
        SELECT
            AVG(sentiment_score)      as avg_sentiment,
            MIN(sentiment_score)      as min_sentiment,
            MAX(sentiment_score)      as max_sentiment,
            SUM(has_injury)           as injury_articles,
            SUM(has_suspension)       as suspension_articles,
            SUM(has_manager_change)   as manager_change,
            SUM(has_crisis)           as crisis_articles,
            SUM(has_positive)         as positive_articles,
            SUM(has_transfer)         as transfer_articles,
            SUM(CASE WHEN sentiment_label='negative' THEN 1 ELSE 0 END) as neg_articles,
            SUM(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END) as pos_articles,
            COUNT(*)                  as total_articles
        FROM news_sentiment
        WHERE team_id = ?
          AND published_at >= ?
          AND published_at <= ?
    """, (team_id, from_date, str(match_date)[:10]))

    row = cur.fetchone()
    if not row or row[0] is None:
        return {k: 0.0 for k in [
            'sentiment_avg', 'sentiment_min', 'sentiment_max',
            'injury_articles', 'suspension_articles', 'manager_change',
            'crisis_articles', 'positive_articles', 'transfer_articles',
            'neg_articles', 'pos_articles', 'total_articles',
        ]}

    return {
        'sentiment_avg':        round(row[0] or 0.0, 4),
        'sentiment_min':        round(row[1] or 0.0, 4),
        'sentiment_max':        round(row[2] or 0.0, 4),
        'injury_articles':      row[3]  or 0,
        'suspension_articles':  row[4]  or 0,
        'manager_change':       min(row[5] or 0, 1),
        'crisis_articles':      row[6]  or 0,
        'positive_articles':    row[7]  or 0,
        'transfer_articles':    row[8]  or 0,
        'neg_articles':         row[9]  or 0,
        'pos_articles':         row[10] or 0,
        'total_articles':       row[11] or 0,
    }


def get_teams_from_db(conn, league_ids=None):
    """Get distinct teams from fixtures table."""
    if league_ids:
        placeholders = ','.join('?' * len(league_ids))
        cur = conn.execute(f"""
            SELECT DISTINCT home_team_id, home_team, league_id
            FROM fixtures
            WHERE league_id IN ({placeholders})
            ORDER BY league_id, home_team
        """, league_ids)
    else:
        cur = conn.execute("""
            SELECT DISTINCT home_team_id, home_team, league_id
            FROM fixtures ORDER BY league_id, home_team
        """)
    return [(r[0], r[1], r[2]) for r in cur.fetchall() if r[0] and r[1]]


def export_parquet(conn, parquet_path):
    df = pd.read_sql(
        "SELECT * FROM news_sentiment ORDER BY published_at DESC", conn)
    df.to_parquet(parquet_path, index=False)
    print(f"\n  Exported {len(df):,} sentiment records → {parquet_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect football news with Twitter-RoBERTa")
    parser.add_argument("--days",         type=int, default=7,
                        help="Days of news to pull (default 7, use 30 for initial load)")
    parser.add_argument("--team",         default=None, help="Single team name")
    parser.add_argument("--league",       type=int, nargs='+', default=None)
    parser.add_argument("--big5",         action="store_true",
                        help="Big 5 leagues only (best English coverage)")
    parser.add_argument("--all-leagues",  action="store_true",
                        help="All 24 leagues (uses multilingual model for non-English)")
    parser.add_argument("--no-classifier",action="store_true",
                        help="Skip zero-shot impact classification (faster)")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip teams already in the DB (default behaviour)")
    parser.add_argument("--rerun",        action="store_true",
                        help="Re-score all articles even if already collected")
    parser.add_argument("--db",           default=None)
    args = parser.parse_args()

    db_path      = args.db or str(DB_PATH)
    parquet_path = str(PARQUET_DIR / "sentiment.parquet")

    conn = get_db(db_path)

    to_date   = datetime.now()
    from_date = to_date - timedelta(days=args.days)

    print(f"\n── News Sentiment Collection (Twitter-RoBERTa) ───────────────")
    print(f"  Device:   {DEVICE.upper()}")
    print(f"  Days:     {args.days}  ({from_date.strftime('%d %b')} → {to_date.strftime('%d %b')})")
    print(f"  Classifier: {'disabled' if args.no_classifier else 'nli-deberta-v3-base'}")

    if args.team:
        teams = [(None, args.team, None)]
    else:
        if args.all_leagues:
            league_ids = [l["id"] for l in LEAGUES]
        elif args.big5:
            league_ids = BIG5_IDS
        elif args.league:
            league_ids = args.league
        else:
            league_ids = BIG5_IDS  # default

        teams = get_teams_from_db(conn, league_ids)

        # Deduplicate by team name
        seen, unique = set(), []
        for tid, tname, lid in teams:
            if tname not in seen:
                seen.add(tname)
                unique.append((tid, tname, lid))
        teams = unique

    # Skip already-collected teams (unless --rerun)
    if not args.rerun:
        existing_ids = set(row[0] for row in
                           conn.execute("SELECT DISTINCT team_id FROM news_sentiment").fetchall())
        before = len(teams)
        teams = [(tid, tname, lid) for tid, tname, lid in teams
                 if tid not in existing_ids]
        skipped = before - len(teams)
        if skipped:
            print(f"  Skipping {skipped} teams already in DB (use --rerun to redo)")

    print(f"  Teams to collect: {len(teams)}\n")

    # Pre-load models before loop (avoids re-loading per team)
    get_sentiment_pipe(multilingual=False)
    if args.all_leagues:
        get_sentiment_pipe(multilingual=True)
    if not args.no_classifier:
        get_classifier_pipe()

    total_articles = 0
    for i, (team_id, team_name, league_id) in enumerate(teams, 1):
        multilingual = is_multilingual_league(league_id or 0)
        print(f"  [{i:3d}/{len(teams)}] {team_name:35s}", end="", flush=True)

        try:
            articles = fetch_team_news(team_name, from_date, to_date,
                                       language='en')

            # For non-English leagues with no English articles, try without language filter
            if not articles and multilingual:
                articles = fetch_team_news(team_name, from_date, to_date,
                                           language='')

            n = process_and_save(conn, articles, team_id, team_name, league_id,
                                 use_classifier=not args.no_classifier)
            total_articles += n

            # Show sample sentiment
            if n > 0:
                avg = sum(
                    float(conn.execute(
                        "SELECT AVG(sentiment_score) FROM news_sentiment WHERE team_name=?",
                        (team_name,)).fetchone()[0] or 0)
                    for _ in [1]
                )
                label = "😊" if avg > 0.15 else ("😟" if avg < -0.15 else "😐")
                print(f" → {n} articles  avg={avg:+.3f} {label}", flush=True)
            else:
                print(f" → 0 articles", flush=True)
                # Save a sentinel row so this team is skipped on restart
                now = __import__('datetime').datetime.utcnow().isoformat()
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO news_sentiment "
                        "(team_id, team_name, league_id, headline, article_url, published_at, "
                        "source_name, sentiment_score, sentiment_engine, collected_at) "
                        "VALUES (?, ?, ?, '__no_articles__', '__no_articles__', ?, "
                        "'__none__', 0.0, '__none__', ?)",
                        (team_id, team_name, league_id, now, now)
                    )
                    conn.commit()
                except Exception as sentinel_err:
                    print(f" [sentinel err: {sentinel_err}]", flush=True)
                    pass

        except Exception as e:
            print(f" ❌ {e}", flush=True)
            continue

    print(f"\n  ✅ Done — {total_articles:,} new articles scored")

    # Summary of impact flags
    try:
        summary = conn.execute("""
            SELECT
                SUM(has_injury)         as injuries,
                SUM(has_suspension)     as suspensions,
                SUM(has_manager_change) as manager_changes,
                SUM(has_crisis)         as crises,
                SUM(has_positive)       as positive,
                SUM(has_transfer)       as transfers,
                COUNT(*)                as total
            FROM news_sentiment
        """).fetchone()
        print(f"\n  Impact breakdown (all time):")
        print(f"  Injuries:        {summary[0]:,}")
        print(f"  Suspensions:     {summary[1]:,}")
        print(f"  Manager changes: {summary[2]:,}")
        print(f"  Team crises:     {summary[3]:,}")
        print(f"  Positive news:   {summary[4]:,}")
        print(f"  Transfers:       {summary[5]:,}")
        print(f"  Total articles:  {summary[6]:,}")
    except Exception:
        pass

    export_parquet(conn, parquet_path)
    conn.close()


if __name__ == "__main__":
    main()
