#!/usr/bin/env python3
"""
build_features.py — Engineer all features for the XGBoost match prediction model.

Takes raw fixtures and team stats from SQLite, outputs a rich feature DataFrame
ready for model training. This is the heart of the prediction pipeline.

Features computed per match (from home team's perspective):
  ── Form (rolling) ──
  - Last 5 results (home/away split + overall): wins, draws, losses, points
  - Goals scored/conceded per game (last 5, last 10)
  - Clean sheets in last 5
  - Goals scored in last 3 (momentum)

  ── Elo Rating ──
  - Home Elo, Away Elo (dynamic, updated after each match)
  - Elo difference (home - away)

  ── Head-to-Head ──
  - H2H last 5 meetings: home wins, draws, away wins
  - H2H average goals
  - H2H at this specific venue (last 3)

  ── Season stats (from team_stats table) ──
  - Goals for/against average (home/away split)
  - Shots on target rate
  - Possession average
  - Pass accuracy

  ── Context ──
  - Rest days (days since last match)
  - Home advantage (always 1 for home team, but useful as sanity check)
  - League division (1 or 2)
  - Match week (early / mid / late season)
  - Derby flag (same-city rivals — hardcoded for known derbies)

  ── Target variables ──
  - result: H / D / A (1X2)
  - result_encoded: 2=H, 1=D, 0=A
  - home_goals, away_goals
  - total_goals, btts (both teams scored)
  - over_2_5 (1 if total_goals > 2)

Usage:
  python3 features/build_features.py
  python3 features/build_features.py --league 39 --season 2024
  python3 features/build_features.py --from-scratch   # recompute all Elo from oldest match
"""

import os, sys, sqlite3, argparse
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from config.settings import DB_PATH, PARQUET_DIR
from config.leagues  import LEAGUES, SEASONS, CURRENT_SEASON, LEAGUE_BY_ID


# ── Elo configuration ──────────────────────────────────────────────────────────

ELO_K         = 32      # K-factor — how much each match shifts Elo
ELO_DEFAULT   = 1500    # Starting Elo for any team with no history
ELO_HOME_ADV  = 100     # Home advantage bonus in expected score calculation


# ── Derby pairs (team IDs for known same-city rivalries) ──────────────────────
# These matches have extra intensity / unpredictability — flag them

DERBY_PAIRS = frozenset([
    frozenset([33, 34]),    # Man United vs Man City
    frozenset([40, 42]),    # Liverpool vs Everton
    frozenset([49, 47]),    # Chelsea vs Arsenal
    frozenset([65, 66]),    # Nottm Forest vs Derby (when both in same league)
    frozenset([529, 530]),  # Barcelona vs Real Madrid (El Clásico)
    frozenset([529, 541]),  # Barcelona vs Atlético Madrid
    frozenset([530, 541]),  # Real Madrid vs Atlético Madrid
    frozenset([489, 487]),  # AC Milan vs Inter (Derby della Madonnina)
    frozenset([496, 488]),  # Juventus vs Torino
    frozenset([85, 91]),    # PSG vs Marseille
    frozenset([159, 165]),  # Borussia Dortmund vs Schalke
    frozenset([157, 168]),  # Bayern München vs 1860 München (when relevant)
])


# ── Elo functions ──────────────────────────────────────────────────────────────

def expected_score(elo_a, elo_b, home_adv=ELO_HOME_ADV):
    """Expected score for team A (home) vs team B (away)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a - home_adv) / 400))


def update_elo(elo_home, elo_away, result, k=ELO_K):
    """
    Update Elo ratings after a match.
    result: 'H' (home win), 'D' (draw), 'A' (away win)
    Returns (new_elo_home, new_elo_away)
    """
    exp_home = expected_score(elo_home, elo_away)
    actual_home = {'H': 1.0, 'D': 0.5, 'A': 0.0}[result]
    actual_away = 1.0 - actual_home

    new_home = elo_home + k * (actual_home - exp_home)
    new_away = elo_away + k * (actual_away - (1.0 - exp_home))
    return round(new_home, 2), round(new_away, 2)


def build_elo_ratings(fixtures_df):
    """
    Process ALL fixtures chronologically to build Elo ratings.
    Returns a dict: {team_id: elo} at any given point + per-fixture pre-match Elo.

    Outputs a DataFrame with columns: fixture_id, pre_elo_home, pre_elo_away
    """
    elo_dict = {}  # team_id -> current Elo

    # Sort by date (then fixture_id as tiebreaker for same-day matches)
    df = fixtures_df.copy()
    df = df.dropna(subset=['result'])  # only completed matches update Elo
    df = df.sort_values(['date', 'fixture_id']).reset_index(drop=True)

    records = []

    for _, row in df.iterrows():
        h_id = row['home_team_id']
        a_id = row['away_team_id']
        result = row['result']

        # Initialise any new teams
        if h_id not in elo_dict:
            elo_dict[h_id] = ELO_DEFAULT
        if a_id not in elo_dict:
            elo_dict[a_id] = ELO_DEFAULT

        pre_elo_h = elo_dict[h_id]
        pre_elo_a = elo_dict[a_id]

        records.append({
            'fixture_id':    row['fixture_id'],
            'pre_elo_home':  pre_elo_h,
            'pre_elo_away':  pre_elo_a,
            'elo_diff':      round(pre_elo_h - pre_elo_a, 2),
        })

        # Update Elo after the match
        new_h, new_a = update_elo(pre_elo_h, pre_elo_a, result)
        elo_dict[h_id] = new_h
        elo_dict[a_id] = new_a

    elo_df = pd.DataFrame(records)
    return elo_df, elo_dict


# ── Rolling form ───────────────────────────────────────────────────────────────

def compute_team_form(fixtures_df, window=5, return_history=False):
    """
    For each match, compute pre-match rolling form for both home and away team.

    Home form: last N matches played AT HOME
    Away form: last N matches played AWAY
    Overall form: last N matches regardless of venue

    Returns DataFrame keyed by fixture_id with form columns.
    """
    df = fixtures_df.copy()
    df = df.dropna(subset=['result'])
    df = df.sort_values(['date', 'fixture_id']).reset_index(drop=True)

    # Build per-team match history
    # Team's perspective: did they win, draw, lose? Goals scored/conceded?
    team_history = {}  # team_id -> list of (date, fixture_id, is_home, scored, conceded, result_for_team)

    for _, row in df.iterrows():
        h_id = row['home_team_id']
        a_id = row['away_team_id']
        if not h_id or not a_id:
            continue

        h_goals = row.get('home_goals', 0) or 0
        a_goals = row.get('away_goals', 0) or 0
        result  = row['result']

        h_result = result           # 'H'=win, 'D'=draw, 'A'=loss  (from home perspective)
        a_result = {'H': 'A', 'D': 'D', 'A': 'H'}[result]

        for team_id, is_home, scored, conceded, team_result in [
            (h_id, True,  h_goals, a_goals, h_result),
            (a_id, False, a_goals, h_goals, a_result),
        ]:
            if team_id not in team_history:
                team_history[team_id] = []
            team_history[team_id].append({
                'date':       row['date'],
                'fixture_id': row['fixture_id'],
                'is_home':    is_home,
                'scored':     scored,
                'conceded':   conceded,
                'result':     team_result,     # W/D/L from this team's view
                'points':     {'H': 3, 'D': 1, 'A': 0}[team_result],
                'clean_sheet': int(conceded == 0),
            })

    def form_stats(matches_subset, n=window):
        """Compute form stats from the last N matches in the subset."""
        recent = matches_subset[-n:]  # already sorted by date
        if not recent:
            return {
                'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
                'goals_scored': 0.0, 'goals_conceded': 0.0, 'clean_sheets': 0,
            }
        n_played = len(recent)
        return {
            'played':          n_played,
            'wins':            sum(1 for m in recent if m['result'] == 'H'),
            'draws':           sum(1 for m in recent if m['result'] == 'D'),
            'losses':          sum(1 for m in recent if m['result'] == 'A'),
            'points':          sum(m['points'] for m in recent),
            'goals_scored':    round(sum(m['scored']    for m in recent) / n_played, 3),
            'goals_conceded':  round(sum(m['conceded']  for m in recent) / n_played, 3),
            'clean_sheets':    sum(m['clean_sheet'] for m in recent),
        }

    # Now build per-fixture pre-match form
    # We use an index pointer approach: for each fixture, we look back in the team's history
    # All matches BEFORE this fixture_id

    # Sort team histories
    for tid in team_history:
        team_history[tid].sort(key=lambda x: (x['date'], x['fixture_id']))

    records = []

    for _, row in df.iterrows():
        fid  = row['fixture_id']
        h_id = row['home_team_id']
        a_id = row['away_team_id']
        if not h_id or not a_id:
            continue

        rec = {'fixture_id': fid}

        for prefix, team_id, venue_flag in [
            ('home_', h_id, True),
            ('away_', a_id, False),
        ]:
            all_prev = [m for m in team_history.get(team_id, [])
                        if m['fixture_id'] < fid]
            home_prev = [m for m in all_prev if m['is_home']]
            away_prev = [m for m in all_prev if not m['is_home']]

            # Overall last 5 and last 10
            overall_5  = form_stats(all_prev,  n=5)
            overall_10 = form_stats(all_prev,  n=10)

            # Venue-specific last 5
            venue_prev  = home_prev if venue_flag else away_prev
            venue_5     = form_stats(venue_prev, n=5)

            # Short momentum: last 3 overall
            momentum_3  = form_stats(all_prev, n=3)

            for k, v in overall_5.items():
                rec[f"{prefix}form5_{k}"] = v
            for k, v in overall_10.items():
                rec[f"{prefix}form10_{k}"] = v
            for k, v in venue_5.items():
                rec[f"{prefix}venue5_{k}"] = v
            for k, v in momentum_3.items():
                rec[f"{prefix}momentum3_{k}"] = v

            # Win rate (avoid div-by-zero)
            n5 = overall_5['played'] or 1
            rec[f"{prefix}win_rate5"] = round(overall_5['wins'] / n5, 3)

        records.append(rec)

    form_df = pd.DataFrame(records)
    if return_history:
        return form_df, team_history
    return form_df


# ── Head-to-head ───────────────────────────────────────────────────────────────

def compute_h2h(fixtures_df, n=5, return_history=False):
    """
    For each match, compute last N H2H meetings between these two teams.
    Returns DataFrame keyed by fixture_id.
    If return_history=True, also returns pair_history dict for upcoming prediction.
    """
    df = fixtures_df.copy()
    df = df.dropna(subset=['result'])
    df = df.sort_values(['date', 'fixture_id']).reset_index(drop=True)

    # Build a lookup: (min_id, max_id) -> sorted list of past meetings
    pair_history = {}  # pair_key -> list of meeting dicts

    for _, row in df.iterrows():
        h_id = row['home_team_id']
        a_id = row['away_team_id']
        if not h_id or not a_id:
            continue
        key = (min(h_id, a_id), max(h_id, a_id))
        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append({
            'fixture_id': row['fixture_id'],
            'date':       row['date'],
            'home_id':    h_id,
            'away_id':    a_id,
            'home_goals': row.get('home_goals', 0) or 0,
            'away_goals': row.get('away_goals', 0) or 0,
            'result':     row['result'],
        })

    records = []

    for _, row in df.iterrows():
        fid  = row['fixture_id']
        h_id = row['home_team_id']
        a_id = row['away_team_id']
        if not h_id or not a_id:
            continue

        key  = (min(h_id, a_id), max(h_id, a_id))
        prev = [m for m in pair_history.get(key, []) if m['fixture_id'] < fid]
        prev_n = prev[-n:]  # last N meetings

        if not prev_n:
            records.append({
                'fixture_id':        fid,
                'h2h_played':        0,
                'h2h_home_wins':     0,
                'h2h_draws':         0,
                'h2h_away_wins':     0,
                'h2h_avg_goals':     0.0,
                'h2h_home_win_rate': 0.0,
            })
            continue

        n_played = len(prev_n)
        home_wins = sum(1 for m in prev_n
                        if (m['home_id'] == h_id and m['result'] == 'H')
                        or (m['away_id'] == h_id and m['result'] == 'A'))
        away_wins = sum(1 for m in prev_n
                        if (m['home_id'] == a_id and m['result'] == 'H')
                        or (m['away_id'] == a_id and m['result'] == 'A'))
        draws     = n_played - home_wins - away_wins
        avg_goals = round(sum(m['home_goals'] + m['away_goals'] for m in prev_n) / n_played, 3)

        records.append({
            'fixture_id':        fid,
            'h2h_played':        n_played,
            'h2h_home_wins':     home_wins,
            'h2h_draws':         draws,
            'h2h_away_wins':     away_wins,
            'h2h_avg_goals':     avg_goals,
            'h2h_home_win_rate': round(home_wins / n_played, 3),
        })

    h2h_df = pd.DataFrame(records)
    if return_history:
        return h2h_df, pair_history
    return h2h_df


# ── Season stats merge ─────────────────────────────────────────────────────────

def merge_season_stats(fixtures_df, team_stats_df):
    """
    Attach team season stats (goals avg, clean sheets rate etc.) to each fixture.
    Uses the team's stats from THAT season (not current season).
    """
    if team_stats_df.empty:
        # Return minimal columns
        return fixtures_df[['fixture_id']].copy()

    ts = team_stats_df[[
        'team_id', 'league_id', 'season',
        'goals_for_avg_total', 'goals_against_avg_total',
        'goals_for_avg_home',  'goals_against_avg_home',
        'goals_for_avg_away',  'goals_against_avg_away',
        'clean_sheets_total',  'played_total',
        'possession_avg',      'pass_accuracy',
        'shots_total',         'shots_on_target',
    ]].copy()

    # Convert avg strings to float (e.g. "1.5")
    for col in ['goals_for_avg_total', 'goals_against_avg_total',
                'goals_for_avg_home', 'goals_against_avg_home',
                'goals_for_avg_away', 'goals_against_avg_away']:
        ts[col] = pd.to_numeric(ts[col], errors='coerce')

    # Clean sheet rate
    ts['clean_sheet_rate'] = (
        ts['clean_sheets_total'] / ts['played_total'].replace(0, np.nan)
    ).infer_objects(copy=False).fillna(0.0).round(3)

    # Shots on target rate
    ts['shots_on_target_rate'] = (
        ts['shots_on_target'] / ts['shots_total'].replace(0, np.nan)
    ).infer_objects(copy=False).fillna(0.0).round(3)

    # Merge for home team
    home_ts = ts.rename(columns={
        c: f"home_ts_{c}" for c in ts.columns if c not in ['team_id', 'league_id', 'season']
    })
    home_ts = home_ts.rename(columns={'team_id': 'home_team_id', 'season': 'season'})
    home_ts = home_ts.drop(columns=['league_id'])

    result = fixtures_df.merge(
        home_ts, on=['home_team_id', 'season'], how='left')

    # Merge for away team
    away_ts = ts.rename(columns={
        c: f"away_ts_{c}" for c in ts.columns if c not in ['team_id', 'league_id', 'season']
    })
    away_ts = away_ts.rename(columns={'team_id': 'away_team_id', 'season': 'season'})
    away_ts = away_ts.drop(columns=['league_id'])

    result = result.merge(away_ts, on=['away_team_id', 'season'], how='left')

    return result


# ── Context features ───────────────────────────────────────────────────────────

def add_context_features(df):
    """Add rest days, match week position, league tier, derby flag."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Division / tier
    df['division']   = df['league_id'].map(lambda x: LEAGUE_BY_ID.get(x, {}).get('div', 1))
    tier_map = {'big5': 3, 'extra_eu': 2, 'americas': 1}
    df['league_tier'] = df['league_id'].map(
        lambda x: tier_map.get(LEAGUE_BY_ID.get(x, {}).get('tier', 'extra_eu'), 2))

    # Derby flag
    df['is_derby'] = df.apply(
        lambda r: int(frozenset([r['home_team_id'], r['away_team_id']]) in DERBY_PAIRS), axis=1)

    # Rest days (days since last match for each team)
    df = df.sort_values('date')
    for prefix, id_col in [('home_', 'home_team_id'), ('away_', 'away_team_id')]:
        last_played = {}
        rest_days_col = f"{prefix}rest_days"
        rest_days = []
        for _, row in df.iterrows():
            tid  = row[id_col]
            dt   = row['date']
            last = last_played.get(tid)
            if last is None or pd.isnull(dt):
                rest_days.append(np.nan)
            else:
                rest_days.append((dt - last).days)
            if not pd.isnull(dt) and tid:
                last_played[tid] = dt
        df[rest_days_col] = rest_days

    # Fill missing rest days with league median (first match of season)
    for col in ['home_rest_days', 'away_rest_days']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Rest advantage
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

    # Season phase: early (< matchday 10), mid (10-30), late (> 30)
    df['match_month'] = df['date'].dt.month.fillna(0).astype(int)
    # August-October = early, Nov-Feb = mid, Mar-May = late
    df['season_phase'] = df['match_month'].map(
        lambda m: 0 if m in [8, 9, 10] else (2 if m in [3, 4, 5, 6] else 1))

    return df


# ── Injury features ────────────────────────────────────────────────────────────

def add_injury_features(df, conn):
    """
    Join injury counts from the injuries table.
    For each match: how many players is each team missing due to injury/suspension?
    Missing key players = massive signal for under-performance.
    """
    # Check if injuries table exists
    try:
        inj_check = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
    except Exception:
        print("  ⚠️  No injuries table — injury features skipped (run collect_injuries.py)")
        df['home_players_out']  = 0
        df['away_players_out']  = 0
        df['home_injuries']     = 0
        df['away_injuries']     = 0
        df['home_suspensions']  = 0
        df['away_suspensions']  = 0
        df['injury_advantage']  = 0
        return df

    if inj_check == 0:
        print("  ⚠️  Injuries table is empty — run collect_injuries.py")
        df['home_players_out']  = 0
        df['away_players_out']  = 0
        df['home_injuries']     = 0
        df['away_injuries']     = 0
        df['home_suspensions']  = 0
        df['away_suspensions']  = 0
        df['injury_advantage']  = 0
        return df

    # Load all injury data and aggregate per fixture per team
    inj_df = pd.read_sql("""
        SELECT fixture_id, team_id,
               SUM(CASE WHEN injury_type='Injury'    THEN 1 ELSE 0 END) as injuries,
               SUM(CASE WHEN injury_type='Suspension' THEN 1 ELSE 0 END) as suspensions,
               COUNT(*) as total_out
        FROM injuries
        WHERE fixture_id IS NOT NULL
        GROUP BY fixture_id, team_id
    """, conn)

    if inj_df.empty:
        print("  ⚠️  No fixture-level injury data available")
        df['home_players_out'] = 0
        df['away_players_out'] = 0
        df['injury_advantage'] = 0
        return df

    # Pivot: home team injuries
    home_inj = inj_df.rename(columns={
        'team_id': 'home_team_id',
        'injuries': 'home_injuries',
        'suspensions': 'home_suspensions',
        'total_out': 'home_players_out',
    })
    away_inj = inj_df.rename(columns={
        'team_id': 'away_team_id',
        'injuries': 'away_injuries',
        'suspensions': 'away_suspensions',
        'total_out': 'away_players_out',
    })

    df = df.merge(home_inj[['fixture_id', 'home_team_id',
                              'home_injuries', 'home_suspensions', 'home_players_out']],
                  on=['fixture_id', 'home_team_id'], how='left')
    df = df.merge(away_inj[['fixture_id', 'away_team_id',
                              'away_injuries', 'away_suspensions', 'away_players_out']],
                  on=['fixture_id', 'away_team_id'], how='left')

    # Fill NaN with 0 (no injury data = assume fully fit)
    for col in ['home_injuries', 'home_suspensions', 'home_players_out',
                'away_injuries', 'away_suspensions', 'away_players_out']:
        df[col] = df[col].fillna(0).astype(int)

    # Derived: negative = home has MORE players out (disadvantage)
    df['injury_advantage'] = df['away_players_out'] - df['home_players_out']

    n_with_data = (df['home_players_out'] > 0).sum() + (df['away_players_out'] > 0).sum()
    print(f"  Injury features added ({n_with_data:,} team-fixtures with data)")

    return df


# ── Sentiment features ─────────────────────────────────────────────────────────

def add_sentiment_features(df, conn):
    """
    Add pre-match news sentiment for each team.
    For each match, aggregate the 7 days of news before kick-off.
    Negative news (injuries, manager sacked, crisis) predicts under-performance.
    Positive news (returns, winning run) predicts over-performance vs. odds.
    """
    try:
        sent_check = conn.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0]
    except Exception:
        print("  ⚠️  No news_sentiment table — run collect_sentiment.py")
        _add_empty_sentiment(df)
        return df

    if sent_check == 0:
        print("  ⚠️  Sentiment table empty — run collect_sentiment.py")
        _add_empty_sentiment(df)
        return df

    # Load all sentiment data, aggregate by team-week
    sent_df = pd.read_sql("""
        SELECT
            team_id,
            strftime('%Y-%W', published_at) as year_week,
            AVG(sentiment_score)   as avg_sentiment,
            MIN(sentiment_score)   as min_sentiment,
            SUM(has_injury)        as injury_articles,
            SUM(has_suspension)    as suspension_articles,
            SUM(has_manager_change) as manager_change,
            SUM(has_crisis)        as crisis_articles,
            SUM(has_positive)      as positive_articles
        FROM news_sentiment
        GROUP BY team_id, year_week
    """, conn)

    if sent_df.empty:
        print("  ⚠️  No sentiment data to merge")
        _add_empty_sentiment(df)
        return df

    # Match each fixture to the sentiment week
    df = df.copy()
    df['year_week'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%W')

    # Merge for home team
    home_sent = sent_df.add_prefix('home_sent_').rename(
        columns={'home_sent_team_id': 'home_team_id', 'home_sent_year_week': 'year_week'})
    df = df.merge(home_sent, on=['home_team_id', 'year_week'], how='left')

    # Merge for away team
    away_sent = sent_df.add_prefix('away_sent_').rename(
        columns={'away_sent_team_id': 'away_team_id', 'away_sent_year_week': 'year_week'})
    df = df.merge(away_sent, on=['away_team_id', 'year_week'], how='left')

    # Fill missing with neutral (0)
    sent_cols = [c for c in df.columns if c.startswith(('home_sent_', 'away_sent_'))]
    for col in sent_cols:
        df[col] = df[col].fillna(0)

    # Derived: sentiment edge (home sentiment - away sentiment)
    if 'home_sent_avg_sentiment' in df.columns and 'away_sent_avg_sentiment' in df.columns:
        df['sentiment_edge'] = df['home_sent_avg_sentiment'] - df['away_sent_avg_sentiment']
    else:
        df['sentiment_edge'] = 0.0

    # Clean up temp column
    df = df.drop(columns=['year_week'], errors='ignore')

    n_with_data = df['home_sent_avg_sentiment'].notna().sum() if 'home_sent_avg_sentiment' in df.columns else 0
    print(f"  Sentiment features added ({n_with_data:,} fixtures with news data)")

    return df


def _add_empty_sentiment(df):
    """Add zero-filled sentiment columns when no data available."""
    for prefix in ['home_sent_', 'away_sent_']:
        df[f'{prefix}avg_sentiment']    = 0.0
        df[f'{prefix}min_sentiment']    = 0.0
        df[f'{prefix}injury_articles']  = 0
        df[f'{prefix}suspension_articles'] = 0
        df[f'{prefix}manager_change']   = 0
        df[f'{prefix}crisis_articles']  = 0
        df[f'{prefix}positive_articles'] = 0
    df['sentiment_edge'] = 0.0


# ── Target encoding ────────────────────────────────────────────────────────────

def add_targets(df):
    """Add all target columns: result encoded, goals, over/under, btts."""
    df = df.copy()
    # 1X2 encoded: 2=home win, 1=draw, 0=away win
    result_map = {'H': 2, 'D': 1, 'A': 0}
    df['result_encoded'] = df['result'].map(result_map)

    # Goals targets
    hg = pd.to_numeric(df['home_goals'], errors='coerce')
    ag = pd.to_numeric(df['away_goals'], errors='coerce')
    df['total_goals'] = hg + ag
    df['btts']        = ((hg > 0) & (ag > 0)).astype(float)
    df['over_2_5']    = (df['total_goals'] > 2).astype(float)
    df['over_1_5']    = (df['total_goals'] > 1).astype(float)
    df['over_3_5']    = (df['total_goals'] > 3).astype(float)

    return df


# ── Upcoming features from historical state ────────────────────────────────────

def build_upcoming_features_from_state(upcoming_df, final_elo, team_history,
                                       pair_history, team_stats_df):
    """
    Build pre-match features for upcoming fixtures using the current state of
    Elo ratings, form, and H2H records computed from all historical data.

    upcoming_df: DataFrame with upcoming fixtures (no result column required)
    final_elo:   dict {team_id: current_elo} — output of build_elo_ratings
    team_history: dict {team_id: [match_dicts]} — from compute_team_form internals
    pair_history: dict {(min_id, max_id): [meeting_dicts]} — from compute_h2h internals
    """
    ELO_DEFAULT_VAL = 1500.0

    def form_stats_from_history(match_list, n=5):
        recent = match_list[-n:]
        if not recent:
            return {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
                    'goals_scored': 0.0, 'goals_conceded': 0.0, 'clean_sheets': 0}
        n_played = len(recent)
        return {
            'played':       n_played,
            'wins':         sum(1 for m in recent if m['result'] == 'H'),
            'draws':        sum(1 for m in recent if m['result'] == 'D'),
            'losses':       sum(1 for m in recent if m['result'] == 'A'),
            'points':       sum(m['points'] for m in recent),
            'goals_scored':  round(sum(m['scored']    for m in recent) / n_played, 3),
            'goals_conceded': round(sum(m['conceded'] for m in recent) / n_played, 3),
            'clean_sheets': sum(1 for m in recent if m['conceded'] == 0),
        }

    records = []
    for _, row in upcoming_df.iterrows():
        fid  = row['fixture_id']
        h_id = row['home_team_id']
        a_id = row['away_team_id']

        pre_elo_h = final_elo.get(h_id, ELO_DEFAULT_VAL)
        pre_elo_a = final_elo.get(a_id, ELO_DEFAULT_VAL)

        rec = {
            'fixture_id':   fid,
            'pre_elo_home': pre_elo_h,
            'pre_elo_away': pre_elo_a,
            'elo_diff':     round(pre_elo_h - pre_elo_a, 2),
        }

        # Form features using all historical matches
        for prefix, team_id, venue_flag in [('home_', h_id, True), ('away_', a_id, False)]:
            all_prev   = team_history.get(team_id, [])
            home_prev  = [m for m in all_prev if m['is_home']]
            away_prev  = [m for m in all_prev if not m['is_home']]
            venue_prev = home_prev if venue_flag else away_prev

            for k, v in form_stats_from_history(all_prev, 5).items():
                rec[f"{prefix}form5_{k}"] = v
            for k, v in form_stats_from_history(all_prev, 10).items():
                rec[f"{prefix}form10_{k}"] = v
            for k, v in form_stats_from_history(venue_prev, 5).items():
                rec[f"{prefix}venue5_{k}"] = v
            for k, v in form_stats_from_history(all_prev, 3).items():
                rec[f"{prefix}momentum3_{k}"] = v

            n5 = form_stats_from_history(all_prev, 5)['played'] or 1
            rec[f"{prefix}win_rate5"] = round(
                form_stats_from_history(all_prev, 5)['wins'] / n5, 3)

        # H2H features
        key = (min(h_id, a_id), max(h_id, a_id))
        past_meetings = pair_history.get(key, [])
        n_met = len(past_meetings)

        if n_met == 0:
            rec.update({'h2h_matches': 0, 'h2h_home_wins': 0, 'h2h_draws': 0,
                        'h2h_away_wins': 0, 'h2h_avg_goals': 0.0,
                        'h2h_home_win_rate': 0.33, 'h2h_draw_rate': 0.33})
        else:
            last5 = past_meetings[-5:]
            home_wins = sum(1 for m in last5 if m['home_id'] == h_id and m['result'] == 'H'
                            or m['away_id'] == h_id and m['result'] == 'A')
            away_wins = sum(1 for m in last5 if m['home_id'] == a_id and m['result'] == 'H'
                            or m['away_id'] == a_id and m['result'] == 'A')
            draws = sum(1 for m in last5 if m['result'] == 'D')
            avg_g = round(np.mean([m['home_goals'] + m['away_goals'] for m in last5]), 3)
            n5_ = len(last5) or 1
            rec.update({
                'h2h_matches':       len(last5),
                'h2h_home_wins':     home_wins,
                'h2h_draws':         draws,
                'h2h_away_wins':     away_wins,
                'h2h_avg_goals':     avg_g,
                'h2h_home_win_rate': round(home_wins / n5_, 3),
                'h2h_draw_rate':     round(draws / n5_, 3),
            })

        records.append(rec)

    features_df = pd.DataFrame(records)

    # Merge back with upcoming fixture info
    result = upcoming_df.merge(features_df, on='fixture_id', how='left')
    result = merge_season_stats(result, team_stats_df)
    result = add_context_features(result)
    return result


# ── Main build ─────────────────────────────────────────────────────────────────

def build_all_features(conn, league_ids=None, seasons=None, include_upcoming=False):
    """
    Full feature build pipeline.
    Returns a DataFrame with all features + targets, one row per completed fixture.

    If include_upcoming=True, also appends feature rows for future scheduled matches
    (no targets — result_encoded will be NaN for those rows).
    """

    # 1. Load completed fixtures
    filters = ["result IS NOT NULL"]
    params  = []
    if league_ids:
        placeholders = ','.join('?' * len(league_ids))
        filters.append(f"league_id IN ({placeholders})")
        params.extend(league_ids)
    if seasons:
        placeholders = ','.join('?' * len(seasons))
        filters.append(f"season IN ({placeholders})")
        params.extend(seasons)

    where = " AND ".join(filters)
    fixtures_df = pd.read_sql(
        f"SELECT * FROM fixtures WHERE {where} ORDER BY date, fixture_id",
        conn, params=params)

    print(f"  Loaded {len(fixtures_df):,} completed fixtures")

    if fixtures_df.empty:
        raise ValueError("No fixtures found. Run collect_fixtures.py first.")

    # 2. Load team stats (for season averages)
    try:
        team_stats_df = pd.read_sql("SELECT * FROM team_stats", conn)
        print(f"  Loaded {len(team_stats_df):,} team-season stat records")
    except Exception:
        team_stats_df = pd.DataFrame()
        print("  ⚠️  No team_stats table — season average features will be empty")

    # 3. Elo ratings
    print("  Computing Elo ratings...")
    elo_df, final_elo = build_elo_ratings(fixtures_df)
    print(f"  Elo computed for {len(final_elo)} teams")

    # 4. Rolling form (return history dict for upcoming features)
    print("  Computing rolling form (last 5 / 10)...")
    form_df, team_history = compute_team_form(fixtures_df, window=5, return_history=True)

    # 5. Head-to-head (return pair_history for upcoming features)
    print("  Computing H2H records...")
    h2h_df, pair_history = compute_h2h(fixtures_df, n=5, return_history=True)

    # 6. Merge everything onto fixtures
    df = fixtures_df.copy()
    df = df.merge(elo_df,  on='fixture_id', how='left')
    df = df.merge(form_df, on='fixture_id', how='left')
    df = df.merge(h2h_df,  on='fixture_id', how='left')

    # 7. Season stats
    print("  Merging season stats...")
    df = merge_season_stats(df, team_stats_df)

    # 8. Context features
    print("  Adding context features...")
    df = add_context_features(df)

    # 9. Injuries
    print("  Adding injury features...")
    df = add_injury_features(df, conn)

    # 10. Sentiment
    print("  Adding sentiment features...")
    df = add_sentiment_features(df, conn)

    # 11. Targets
    df = add_targets(df)

    # 12. Filter to rows with valid targets (completed matches with valid result)
    df = df[df['result_encoded'].notna()].copy()
    df['result_encoded'] = df['result_encoded'].astype(int)

    # 13. If requested, also build features for upcoming fixtures
    if include_upcoming:
        upcoming_sql_params = []
        upcoming_filters    = ["result IS NULL", "date >= date('now')"]
        if league_ids:
            placeholders = ','.join('?' * len(league_ids))
            upcoming_filters.append(f"league_id IN ({placeholders})")
            upcoming_sql_params.extend(league_ids)

        upcoming_df = pd.read_sql(
            f"SELECT * FROM fixtures WHERE {' AND '.join(upcoming_filters)} ORDER BY date, fixture_id",
            conn, params=upcoming_sql_params)

        if not upcoming_df.empty:
            print(f"\n  Building features for {len(upcoming_df):,} upcoming fixtures...")
            up_features = build_upcoming_features_from_state(
                upcoming_df, final_elo, team_history, pair_history, team_stats_df)
            up_features = add_injury_features(up_features, conn)
            up_features = add_sentiment_features(up_features, conn)
            up_features = add_targets(up_features)   # sets targets to NaN for upcoming
            # Align columns — ensure up_features has same schema as df
            for col in df.columns:
                if col not in up_features.columns:
                    # Use correct dtype to avoid concat FutureWarning
                    up_features[col] = (
                        pd.array([np.nan] * len(up_features), dtype=df[col].dtype)
                        if pd.api.types.is_float_dtype(df[col])
                        else None
                    )
            up_features = up_features[[c for c in df.columns if c in up_features.columns]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                df = pd.concat([df, up_features], ignore_index=True)
            print(f"  {len(up_features):,} upcoming matches added")

    print(f"\n  ✅ Feature build complete: {len(df):,} rows × {len(df.columns)} columns")

    return df


def get_feature_columns(df):
    """
    Return the list of feature columns for model training.
    Excludes raw/admin columns and target columns.
    """
    EXCLUDE_PREFIXES = ['raw_', 'collected_']
    EXCLUDE_EXACT    = {
        'fixture_id', 'league_id', 'league_name', 'country', 'season',
        'date', 'timestamp', 'status', 'home_team_id', 'away_team_id',
        'home_team', 'away_team', 'home_goals', 'away_goals',
        'home_ht', 'away_ht', 'venue', 'referee',
        'result',
        # Targets (exclude from features):
        'result_encoded', 'total_goals', 'btts', 'over_2_5', 'over_1_5', 'over_3_5',
    }

    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_EXACT:
            continue
        if any(col.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        # Must be numeric
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            feature_cols.append(col)

    return feature_cols


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build features for the football model")
    parser.add_argument("--league",   type=int,  nargs='+', default=None,
                        help="Filter to specific league IDs")
    parser.add_argument("--season",   type=int,  nargs='+', default=None,
                        help="Filter to specific seasons")
    parser.add_argument("--upcoming", action="store_true",
                        help="Also build features for upcoming scheduled matches")
    parser.add_argument("--db",       default=None)
    parser.add_argument("--output",   default=None,
                        help="Output parquet path (default: data/parquet/features.parquet)")
    args = parser.parse_args()

    db_path     = args.db     or str(DB_PATH)
    output_path = args.output or str(PARQUET_DIR / "features.parquet")

    conn = sqlite3.connect(db_path)

    print(f"\n── Building features ────────────────────────────────────────")
    df = build_all_features(
        conn,
        league_ids=args.league,
        seasons=args.season,
        include_upcoming=args.upcoming,
    )

    feature_cols = get_feature_columns(df)
    print(f"  Feature columns: {len(feature_cols)}")

    df.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")

    # Print feature column list
    print(f"\n  ── Feature columns ({'showing first 40'})")
    for fc in feature_cols[:40]:
        non_null = df[fc].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"  {fc:50s}  {pct:5.1f}% non-null")
    if len(feature_cols) > 40:
        print(f"  ... and {len(feature_cols) - 40} more")

    conn.close()


if __name__ == "__main__":
    main()
