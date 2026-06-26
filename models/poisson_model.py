#!/usr/bin/env python3
"""
poisson_model.py — Dixon-Coles Poisson regression for scoreline prediction.

The Dixon-Coles model is the gold standard for football scoreline prediction.
It models goals as Poisson processes with:
  - Attack strength per team (how many goals they score)
  - Defence weakness per team (how many goals they concede)
  - Home advantage parameter
  - Low-score correction (0-0, 1-0, 0-1, 1-1 are under/over-represented vs Poisson)
  - Time-decay weighting (recent matches weighted more than old ones)

Output per match:
  - Full scoreline distribution (P(0-0), P(1-0), P(0-1), ... up to 6-6)
  - Home win / Draw / Away win probabilities (sum of scoreline probs)
  - Expected goals (home and away)
  - Most likely scoreline

Time-decay: matches from 5+ years ago are weighted near-zero.
Re-fit: should be re-run after each matchday for live attack/defence ratings.

Usage:
  python3 models/poisson_model.py --train
  python3 models/poisson_model.py --train --league 39
  python3 models/poisson_model.py --predict --home "Arsenal" --away "Chelsea"
  python3 models/poisson_model.py --predict-upcoming  # use upcoming fixtures
"""

import os, sys, json, sqlite3, argparse, pickle
from datetime import timezone, datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from config.settings import DB_PATH, PARQUET_DIR

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved')
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Dixon-Coles correction ─────────────────────────────────────────────────────

def dixon_coles_tau(x, y, lambda_home, mu_away, rho):
    """
    Low-score correction factor (Dixon-Coles 1997).
    Corrects the Poisson assumption for scores of 0-0, 1-0, 0-1, 1-1.
    """
    if   x == 0 and y == 0: return 1.0 - lambda_home * mu_away * rho
    elif x == 1 and y == 0: return 1.0 + mu_away  * rho
    elif x == 0 and y == 1: return 1.0 + lambda_home * rho
    elif x == 1 and y == 1: return 1.0 - rho
    else:                    return 1.0


def time_weight(match_date, ref_date, xi=0.0018):
    """
    Exponential time decay weight.
    xi=0.0018 → half-life ≈ 385 days (roughly one season)
    Matches older than 3 seasons get weight < 0.14
    """
    if pd.isnull(match_date):
        return 0.1
    days = max(0, (ref_date - match_date).days)
    return np.exp(-xi * days)


# ── Vectorised negative log-likelihood ────────────────────────────────────────

def _prepare_fixtures_arrays(fixtures_df, teams, ref_date, xi=0.0018):
    """
    Pre-compute integer arrays for vectorised NLL.
    Called once before optimisation; result is passed via args.

    Returns (hi_arr, ai_arr, hg_arr, ag_arr, weights) as numpy arrays.
    """
    team_idx = {t: i for i, t in enumerate(teams)}

    hi_list, ai_list, hg_list, ag_list, w_list = [], [], [], [], []

    for _, row in fixtures_df.iterrows():
        hi = team_idx.get(row['home_team_id'])
        ai = team_idx.get(row['away_team_id'])
        if hi is None or ai is None:
            continue
        hg = row['home_goals']
        ag = row['away_goals']
        if pd.isnull(hg) or pd.isnull(ag):
            continue
        w = time_weight(
            pd.to_datetime(row['date'], errors='coerce'),
            ref_date, xi)
        hi_list.append(hi);  ai_list.append(ai)
        hg_list.append(int(hg)); ag_list.append(int(ag))
        w_list.append(w)

    return (np.array(hi_list,  dtype=np.int32),
            np.array(ai_list,  dtype=np.int32),
            np.array(hg_list,  dtype=np.int32),
            np.array(ag_list,  dtype=np.int32),
            np.array(w_list,   dtype=np.float64))


def neg_log_likelihood(params, n, hi_arr, ai_arr, hg_arr, ag_arr, weights):
    """
    Dixon-Coles NLL — fully vectorised over fixtures (no Python for-loop).

    params layout:
      [0..N-1]    attack_i
      [N..2N-1]   defence_i
      [2N]        home_adv
      [2N+1]      rho
    """
    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    if abs(rho) > 2.0:
        return 1e9

    # Expected goals per fixture (vectorised)
    lam = np.exp(home_adv + attack[hi_arr] - defence[ai_arr])  # home xG
    mu  = np.exp(attack[ai_arr] - defence[hi_arr])              # away xG

    # Poisson log-PMF: log(λ^k * e^{-λ} / k!) = k*log(λ) - λ - logΓ(k+1)
    from scipy.special import gammaln
    log_p_h = hg_arr * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(hg_arr + 1)
    log_p_a = ag_arr * np.log(np.maximum(mu,  1e-10)) - mu  - gammaln(ag_arr + 1)

    # Dixon-Coles tau correction (vectorised per case)
    tau = np.ones(len(hg_arr), dtype=np.float64)
    m00 = (hg_arr == 0) & (ag_arr == 0)
    m10 = (hg_arr == 1) & (ag_arr == 0)
    m01 = (hg_arr == 0) & (ag_arr == 1)
    m11 = (hg_arr == 1) & (ag_arr == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho

    tau = np.maximum(tau, 1e-10)

    log_p = log_p_h + log_p_a + np.log(tau)
    ll = np.dot(weights, log_p)

    return -ll


# ── Fit model ─────────────────────────────────────────────────────────────────

def fit_poisson(fixtures_df, ref_date=None, xi=0.0018):
    """
    Fit Dixon-Coles Poisson model to completed fixtures.

    Returns dict:
      teams, attack, defence, home_adv, rho, ref_date, fitted_at
    """
    if ref_date is None:
        ref_date = datetime.now()

    df = fixtures_df.dropna(subset=['home_goals', 'away_goals', 'result']).copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Teams present in this dataset
    home_teams = set(df['home_team_id'].dropna())
    away_teams = set(df['away_team_id'].dropna())
    teams = sorted(home_teams | away_teams)
    n = len(teams)

    if n == 0:
        raise ValueError("No teams found in fixtures")

    print(f"  Fitting Poisson model: {n} teams, {len(df):,} matches")

    # Initial parameters: attack=0, defence=0, home_adv=0.1, rho=-0.1
    params0 = np.zeros(2 * n + 2)
    params0[2*n]     =  0.10   # home advantage
    params0[2*n + 1] = -0.10   # rho (negative = correction for low-score games)

    # Pre-compute fixture arrays (avoids Python for-loop inside optimiser)
    print("  Pre-computing fixture arrays...")
    hi_arr, ai_arr, hg_arr, ag_arr, weights = _prepare_fixtures_arrays(
        df, teams, ref_date, xi)
    print(f"  Usable fixtures after filtering: {len(hi_arr):,}")

    # maxiter scales with team count: ~4 per parameter (2N+2 params)
    max_iter = max(1000, 4 * (2 * n + 2))
    print(f"  Running optimisation (vectorised, maxiter={max_iter})...")

    result = minimize(
        neg_log_likelihood,
        params0,
        args=(n, hi_arr, ai_arr, hg_arr, ag_arr, weights),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5},
    )

    if not result.success:
        print(f"  ⚠️  Optimiser warning: {result.message}")
        print(f"      (Parameters still valid — using best found values)")

    params = result.x
    attack  = {t: params[i]     for i, t in enumerate(teams)}
    defence = {t: params[n + i] for i, t in enumerate(teams)}
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    print(f"  Home advantage: {home_adv:.4f} (exp={np.exp(home_adv):.3f}x)")
    print(f"  Rho (DC correction): {rho:.4f}")
    print(f"  Optimiser converged: {result.success}")

    return {
        'teams':     teams,
        'attack':    attack,
        'defence':   defence,
        'home_adv':  home_adv,
        'rho':       rho,
        'xi':        xi,
        'ref_date':  ref_date.isoformat(),
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'n_matches': len(df),
    }


# ── Predict scoreline ──────────────────────────────────────────────────────────

def predict_scoreline(model, home_team_id, away_team_id, max_goals=6):
    """
    Generate full scoreline probability distribution.

    Returns dict with:
      - scoreline_probs: {(home, away): probability}
      - prob_home, prob_draw, prob_away
      - expected_home, expected_away
      - most_likely_scoreline: (h, a)
    """
    attack  = model['attack']
    defence = model['defence']
    home_adv = model['home_adv']
    rho     = model['rho']

    h_att = attack.get(home_team_id)
    h_def = defence.get(home_team_id)
    a_att = attack.get(away_team_id)
    a_def = defence.get(away_team_id)

    if any(x is None for x in [h_att, h_def, a_att, a_def]):
        # Unknown team — use league average (0.0)
        h_att = h_att or 0.0
        h_def = h_def or 0.0
        a_att = a_att or 0.0
        a_def = a_def or 0.0

    lam = np.exp(home_adv + h_att - a_def)   # home expected goals
    mu  = np.exp(a_att - h_def)               # away expected goals

    # Full scoreline matrix
    scoreline_probs = {}
    total = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tau = dixon_coles_tau(h, a, lam, mu, rho)
            p   = poisson.pmf(h, lam) * poisson.pmf(a, mu) * tau
            p   = max(p, 0.0)
            scoreline_probs[(h, a)] = p
            total += p

    # Normalise (DC correction means probs don't exactly sum to 1)
    if total > 0:
        scoreline_probs = {k: v / total for k, v in scoreline_probs.items()}

    # Aggregate outcomes
    prob_home = sum(v for (h, a), v in scoreline_probs.items() if h > a)
    prob_draw = sum(v for (h, a), v in scoreline_probs.items() if h == a)
    prob_away = sum(v for (h, a), v in scoreline_probs.items() if h < a)

    most_likely = max(scoreline_probs, key=scoreline_probs.get)

    return {
        'scoreline_probs':      {f"{h}-{a}": round(v, 5)
                                 for (h, a), v in scoreline_probs.items()},
        'prob_home':            round(prob_home, 4),
        'prob_draw':            round(prob_draw, 4),
        'prob_away':            round(prob_away, 4),
        'expected_home_goals':  round(lam, 3),
        'expected_away_goals':  round(mu,  3),
        'most_likely_scoreline': f"{most_likely[0]}-{most_likely[1]}",
        'p_most_likely':        round(scoreline_probs[most_likely], 4),
    }


def predict_upcoming_fixtures(model, upcoming_df):
    """
    Run Poisson predictions for all upcoming fixtures.
    Returns DataFrame with predictions.
    """
    rows = []
    for _, row in upcoming_df.iterrows():
        pred = predict_scoreline(model, row['home_team_id'], row['away_team_id'])
        pred['fixture_id'] = row.get('fixture_id')
        pred['home_team']  = row.get('home_team')
        pred['away_team']  = row.get('away_team')
        pred['date']       = row.get('date')
        pred['league_id']  = row.get('league_id')
        rows.append(pred)

    df = pd.DataFrame(rows)

    # Reorder columns
    front = ['fixture_id', 'home_team', 'away_team', 'date',
             'prob_home', 'prob_draw', 'prob_away',
             'expected_home_goals', 'expected_away_goals',
             'most_likely_scoreline', 'p_most_likely']
    other = [c for c in df.columns if c not in front]
    return df[[c for c in front if c in df.columns] + other]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poisson scoreline model")
    parser.add_argument('--train',             action='store_true',
                        help='Fit Poisson model from fixtures in DB')
    parser.add_argument('--predict-upcoming',  action='store_true',
                        help='Predict upcoming fixtures (need fixtures.parquet)')
    parser.add_argument('--predict',           action='store_true',
                        help='Predict a single match (use --home and --away)')
    parser.add_argument('--home',              default=None,
                        help='Home team ID (integer) for single prediction')
    parser.add_argument('--away',              default=None,
                        help='Away team ID (integer) for single prediction')
    parser.add_argument('--league',            type=int, nargs='+', default=None)
    parser.add_argument('--max-goals',         type=int, default=6)
    parser.add_argument('--xi',                type=float, default=0.0018,
                        help='Time-decay rate (default 0.0018 = ~1 season half-life)')
    parser.add_argument('--db',                default=None)
    args = parser.parse_args()

    db_path    = args.db or str(DB_PATH)
    model_path = os.path.join(MODEL_DIR, 'poisson_model.pkl')

    if args.train:
        conn = sqlite3.connect(db_path)

        # Load completed fixtures
        filters = ["result IS NOT NULL", "home_goals IS NOT NULL"]
        sql_params = []
        if args.league:
            placeholders = ','.join('?' * len(args.league))
            filters.append(f"league_id IN ({placeholders})")
            sql_params.extend(args.league)

        df = pd.read_sql(
            f"SELECT * FROM fixtures WHERE {' AND '.join(filters)} ORDER BY date",
            conn, params=sql_params or None)
        conn.close()

        print(f"\n── Fitting Poisson models (per-league) ──────────────────────")
        print(f"  Total fixtures: {len(df):,}")

        # Dixon-Coles is per-league: attack/defence ratings only meaningful within league
        league_ids = sorted(df['league_id'].unique()) if not args.league else args.league
        all_models = {}
        ref_date   = datetime.now()

        for lid in league_ids:
            league_df = df[df['league_id'] == lid].copy()
            n_teams   = league_df[['home_team_id','away_team_id']].stack().nunique()
            n_matches = len(league_df)
            league_name = league_df['league_name'].iloc[0] if 'league_name' in league_df.columns else str(lid)
            print(f"\n  {league_name} (id={lid}): {n_teams} teams, {n_matches:,} matches")
            try:
                m = fit_poisson(league_df, ref_date=ref_date, xi=args.xi)
                all_models[lid] = m
                # Top 5 attackers
                top5 = sorted(m['attack'].items(), key=lambda x: x[1], reverse=True)[:5]
                names = league_df.set_index('home_team_id')['home_team'].to_dict()
                for tid, att in top5:
                    name = names.get(tid, str(tid))
                    print(f"    {name:30s}: attack={att:+.3f} ({np.exp(att):.2f}x)")
            except Exception as e:
                print(f"  ❌ {e}")

        print(f"\n  ✅ Fitted {len(all_models)} league models")

        # Save all league models in one pkl (keyed by league_id)
        with open(model_path, 'wb') as f:
            pickle.dump(all_models, f)
        print(f"  Saved → {model_path}")

        # Also save per-league for easy loading
        league_dir = os.path.join(MODEL_DIR, 'poisson_leagues')
        os.makedirs(league_dir, exist_ok=True)
        for lid, m in all_models.items():
            with open(os.path.join(league_dir, f'poisson_{lid}.pkl'), 'wb') as f:
                pickle.dump(m, f)
        print(f"  Per-league models → {league_dir}/")

    elif args.predict:
        if not args.home or not args.away:
            print("❌ Provide --home TEAM_ID and --away TEAM_ID")
            sys.exit(1)
        if not args.league or len(args.league) != 1:
            print("❌ Provide --league LEAGUE_ID for single-match prediction")
            sys.exit(1)

        if not os.path.exists(model_path):
            print("❌ No trained model — run --train first")
            sys.exit(1)

        with open(model_path, 'rb') as f:
            all_models = pickle.load(f)

        lid   = args.league[0]
        model = all_models.get(lid)
        if model is None:
            print(f"❌ No Poisson model for league {lid}")
            sys.exit(1)

        pred = predict_scoreline(model, int(args.home), int(args.away),
                                  max_goals=args.max_goals)

        print(f"\n  {'Home':25s}  vs  {'Away':25s}")
        print(f"  {'─'*60}")
        print(f"  Home win:  {pred['prob_home']:.1%}")
        print(f"  Draw:      {pred['prob_draw']:.1%}")
        print(f"  Away win:  {pred['prob_away']:.1%}")
        print(f"  xG home:   {pred['expected_home_goals']}")
        print(f"  xG away:   {pred['expected_away_goals']}")
        print(f"  Most likely: {pred['most_likely_scoreline']} ({pred['p_most_likely']:.1%})")
        print(f"\n  Top 10 most likely scorelines:")
        top10 = sorted(pred['scoreline_probs'].items(),
                       key=lambda x: x[1], reverse=True)[:10]
        for score, p in top10:
            bar = '█' * int(p * 200)
            print(f"    {score:5s}  {p:.2%}  {bar}")

    elif args.predict_upcoming:
        if not os.path.exists(model_path):
            print("❌ No trained model — run --train first")
            sys.exit(1)

        with open(model_path, 'rb') as f:
            all_models = pickle.load(f)

        fixtures_path = str(PARQUET_DIR / "fixtures.parquet")
        if not os.path.exists(fixtures_path):
            print("❌ No fixtures parquet — run collect_fixtures.py first")
            sys.exit(1)

        df = pd.read_parquet(fixtures_path)
        upcoming = df[df['result'].isna()].copy()
        print(f"  Upcoming fixtures: {len(upcoming):,}")

        all_preds = []
        for lid, model in all_models.items():
            league_upcoming = upcoming[upcoming['league_id'] == lid]
            if len(league_upcoming) == 0:
                continue
            preds = predict_upcoming_fixtures(model, league_upcoming)
            all_preds.append(preds)

        if not all_preds:
            print("  No upcoming fixtures found in any league")
        else:
            predictions = pd.concat(all_preds, ignore_index=True)
            out_path = str(PARQUET_DIR / "predictions_poisson.parquet")
            predictions.to_parquet(out_path, index=False)
            print(f"  Predictions saved → {out_path}")
            print(predictions[['home_team', 'away_team', 'date',
                                'prob_home', 'prob_draw', 'prob_away',
                                'most_likely_scoreline']].head(20).to_string(index=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
