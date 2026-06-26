#!/usr/bin/env python3
"""
national_team_model.py — World Cup / international prediction model.

Two components fitted on intl_fixtures (senior national-team results):
  1. Elo ratings  — chronological, competition-weighted, goal-difference adjusted.
  2. Dixon-Coles Poisson — attack/defence per nation + home advantage (neutral-aware),
     time-weighted. Gives scoreline distribution → 1X2 / Over-Under 2.5 / BTTS.

The final 1X2 is a blend of Poisson and Elo (robustness on sparse intl data).

Usage:
  source .env
  python3 models/national_team_model.py --train
  python3 models/national_team_model.py --predict-wc          # upcoming WC 2026 fixtures
  python3 models/national_team_model.py --predict --home "France" --away "Norway" [--neutral]
"""

import os, sys, json, pickle, argparse, sqlite3, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln

# Youth / women / olympic / futsal sides — excluded so only senior men's teams are rated
JUNK_TEAM = re.compile(r"(\bU-?\d{2}\b|\bU\d{2}|Women|\bW\b|Olympic|Futsal|Beach)", re.I)

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "national_team_model.pkl")

# Elo K-factor by competition tier (higher = match matters more)
ELO_K = {
    1: 60,    # World Cup
    4: 50, 9: 50, 6: 50, 7: 50, 22: 40,        # continental finals
    5: 40, 536: 35,                             # nations leagues
    32: 35, 34: 35, 29: 35, 30: 35, 31: 35, 33: 30, 37: 40,  # WC qualifiers
    10: 20,   # friendlies
}
ELO_HOME_ADV = 65.0   # Elo points for a genuine home tie (non-neutral)


def load_intl(conn, include_unplayed=False):
    cond = "" if include_unplayed else "result IS NOT NULL AND"
    df = pd.read_sql_query(
        f"SELECT * FROM intl_fixtures WHERE {cond} "
        "home_team_id IS NOT NULL AND away_team_id IS NOT NULL", conn)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    # drop youth / women / olympic / futsal sides
    mask = ~(df['home_team'].fillna('').apply(lambda s: bool(JUNK_TEAM.search(s))) |
             df['away_team'].fillna('').apply(lambda s: bool(JUNK_TEAM.search(s))))
    df = df[mask]
    if not include_unplayed:
        df = df.dropna(subset=['home_goals', 'away_goals'])
    return df.reset_index(drop=True)


# ── Elo ─────────────────────────────────────────────────────────────────────
def build_elo(df, base=1500.0):
    rating = {}
    names = {}
    for r in df.itertuples():
        h, a = r.home_team_id, r.away_team_id
        names[h], names[a] = r.home_team, r.away_team
        Rh = rating.get(h, base); Ra = rating.get(a, base)
        ha = 0.0 if r.neutral else ELO_HOME_ADV
        Eh = 1.0 / (1.0 + 10 ** (-((Rh + ha) - Ra) / 400.0))
        # actual result
        if r.home_goals > r.away_goals:   Sh = 1.0
        elif r.home_goals < r.away_goals: Sh = 0.0
        else:                              Sh = 0.5
        # goal-difference multiplier (World Football Elo style)
        gd = abs(r.home_goals - r.away_goals)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))
        K = ELO_K.get(r.league_id, 25) * g
        rating[h] = Rh + K * (Sh - Eh)
        rating[a] = Ra + K * ((1.0 - Sh) - (1.0 - Eh))
    return rating, names


def elo_win_probs(model, home_id, away_id, neutral):
    """W/D/L from Elo. Draw share scaled by closeness."""
    Rh = model['elo'].get(home_id, 1500.0)
    Ra = model['elo'].get(away_id, 1500.0)
    ha = 0.0 if neutral else ELO_HOME_ADV
    Eh = 1.0 / (1.0 + 10 ** (-((Rh + ha) - Ra) / 400.0))   # 2-way expected
    # empirical draw model: max ~28% at perfectly even, shrinks with mismatch
    p_draw = 0.28 * (1.0 - abs(2 * Eh - 1)) + 0.16 * abs(2 * Eh - 1) * 0  # ~0.28..0
    p_draw = max(0.06, 0.28 * (1.0 - abs(2 * Eh - 1)))
    p_home = Eh * (1 - p_draw)
    p_away = (1 - Eh) * (1 - p_draw)
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


# ── Dixon-Coles Poisson (neutral-aware, time-weighted) ───────────────────────
def _arrays(df, teams, ref_date, xi):
    idx = {t: i for i, t in enumerate(teams)}
    hi = df['home_team_id'].map(idx).to_numpy()
    ai = df['away_team_id'].map(idx).to_numpy()
    hg = df['home_goals'].to_numpy(dtype=np.int32)
    ag = df['away_goals'].to_numpy(dtype=np.int32)
    neu = df['neutral'].fillna(0).to_numpy(dtype=np.float64)
    days = (ref_date - df['date']).dt.days.to_numpy(dtype=np.float64)
    w = np.exp(-xi * np.maximum(days, 0))
    return hi, ai, hg, ag, neu, w


def _nll(params, n, hi, ai, hg, ag, neu, w):
    attack = params[:n]; defence = params[n:2*n]
    home_adv = params[2*n]; rho = params[2*n + 1]
    if abs(rho) > 2.0:
        return 1e10
    lam = np.exp(home_adv * (1 - neu) + attack[hi] - defence[ai])
    mu  = np.exp(attack[ai] - defence[hi])
    lam = np.clip(lam, 1e-10, 30); mu = np.clip(mu, 1e-10, 30)
    log_h = hg * np.log(lam) - lam - gammaln(hg + 1)
    log_a = ag * np.log(mu)  - mu  - gammaln(ag + 1)
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0); m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1); m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)
    return -np.dot(w, log_h + log_a + np.log(tau))


def fit(df, xi=0.0010):   # xi tuned via time-based backtest (best log-loss/accuracy)
    ref = df['date'].max()
    teams = sorted(set(df['home_team_id']) | set(df['away_team_id']))
    n = len(teams)
    hi, ai, hg, ag, neu, w = _arrays(df, teams, ref, xi)
    p0 = np.zeros(2 * n + 2); p0[2*n] = 0.25; p0[2*n + 1] = -0.10
    print(f"  Fitting Dixon-Coles: {n} nations, {len(df):,} matches (ref {ref.date()})")
    res = minimize(_nll, p0, args=(n, hi, ai, hg, ag, neu, w),
                   method='L-BFGS-B', options={'maxiter': max(2000, 8*n), 'ftol': 1e-9})
    p = res.x
    elo, names = build_elo(df)
    print(f"  Home advantage: {p[2*n]:.3f} (exp={np.exp(p[2*n]):.3f}x) | rho={p[2*n+1]:.3f} | converged={res.success}")
    return {
        'teams': teams, 'attack': {t: p[i] for i, t in enumerate(teams)},
        'defence': {t: p[n+i] for i, t in enumerate(teams)},
        'home_adv': float(p[2*n]), 'rho': float(p[2*n+1]), 'xi': xi,
        'elo': elo, 'names': names,
        'name_to_id': {v: k for k, v in names.items()},
        'ref_date': ref.isoformat(), 'fitted_at': datetime.now(timezone.utc).isoformat(),
        'n_matches': len(df),
    }


def _dc_tau(h, a, lam, mu, rho):
    if h == 0 and a == 0: return 1 - lam * mu * rho
    if h == 1 and a == 0: return 1 + mu * rho
    if h == 0 and a == 1: return 1 + lam * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0


def predict(model, home_id, away_id, neutral=True, max_goals=10, blend=1.0):  # blend=1.0: Poisson-primary (best calibrated in backtest); Elo shown as cross-check
    att, dfc = model['attack'], model['defence']
    ha = model['home_adv'] * (0 if neutral else 1)
    lam = np.exp(ha + att.get(home_id, 0.0) - dfc.get(away_id, 0.0))
    mu  = np.exp(att.get(away_id, 0.0) - dfc.get(home_id, 0.0))
    rho = model['rho']
    M = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            M[h, a] = poisson.pmf(h, lam) * poisson.pmf(a, mu) * _dc_tau(h, a, lam, mu, rho)
    M = np.maximum(M, 0); M /= M.sum()
    p_home = np.tril(M, -1).sum(); p_draw = np.trace(M); p_away = np.triu(M, 1).sum()
    over25 = sum(M[h, a] for h in range(max_goals+1) for a in range(max_goals+1) if h + a > 2.5)
    btts = sum(M[h, a] for h in range(1, max_goals+1) for a in range(1, max_goals+1))
    flat = [((h, a), M[h, a]) for h in range(max_goals+1) for a in range(max_goals+1)]
    flat.sort(key=lambda x: -x[1])
    # Elo blend on 1X2
    eh, ed, ea = elo_win_probs(model, home_id, away_id, neutral)
    bh = blend * p_home + (1 - blend) * eh
    bd = blend * p_draw + (1 - blend) * ed
    ba = blend * p_away + (1 - blend) * ea
    s = bh + bd + ba
    return {
        'lambda_home': round(lam, 2), 'lambda_away': round(mu, 2),
        'poisson_1x2': (round(p_home, 3), round(p_draw, 3), round(p_away, 3)),
        'elo_1x2': (round(eh, 3), round(ed, 3), round(ea, 3)),
        'blend_1x2': (round(bh/s, 3), round(bd/s, 3), round(ba/s, 3)),
        'over_2_5': round(float(over25), 3), 'under_2_5': round(1 - float(over25), 3),
        'btts_yes': round(float(btts), 3), 'btts_no': round(1 - float(btts), 3),
        'top_scores': [(f"{h}-{a}", round(p, 3)) for (h, a), p in flat[:4]],
        'elo_home': round(model['elo'].get(home_id, 1500)),
        'elo_away': round(model['elo'].get(away_id, 1500)),
    }


def _poisson_elo_1x2(model, home_id, away_id, neutral, max_goals=10):
    """Return (poisson_1x2, elo_1x2) without blending — for fast tuning."""
    att, dfc = model['attack'], model['defence']
    ha = model['home_adv'] * (0 if neutral else 1)
    lam = np.exp(ha + att.get(home_id, 0.0) - dfc.get(away_id, 0.0))
    mu  = np.exp(att.get(away_id, 0.0) - dfc.get(home_id, 0.0))
    rho = model['rho']
    hh = poisson.pmf(np.arange(max_goals+1), lam)
    aa = poisson.pmf(np.arange(max_goals+1), mu)
    M = np.outer(hh, aa)
    M[0,0] *= 1 - lam*mu*rho; M[1,0] *= 1 + mu*rho
    M[0,1] *= 1 + lam*rho;    M[1,1] *= 1 - rho
    M = np.maximum(M, 0); M /= M.sum()
    ph = np.tril(M,-1).sum(); pdr = np.trace(M); pa = np.triu(M,1).sum()
    return (ph, pdr, pa), elo_win_probs(model, home_id, away_id, neutral)


def tune(df, cutoff="2025-06-01"):
    """Time-based backtest: fit on date<cutoff, score 1X2 on recent competitive matches."""
    cutoff = pd.Timestamp(cutoff)
    train = df[df['date'] < cutoff]
    test  = df[(df['date'] >= cutoff) & (df['league_id'] != 10)]   # exclude friendlies from test
    print(f"\n  Backtest: train {len(train):,} (<{cutoff.date()}) | test {len(test):,} competitive matches")
    grid_xi   = [0.0006, 0.0010, 0.0015, 0.0020]
    grid_blend = [0.0, 0.3, 0.5, 0.7, 1.0]   # 0=Elo only, 1=Poisson only
    out = []
    for xi in grid_xi:
        m = fit(train, xi=xi)
        seen = set(m['teams'])
        sub = test[test['home_team_id'].isin(seen) & test['away_team_id'].isin(seen)]
        # precompute poisson+elo per match once
        cache = []
        for r in sub.itertuples():
            (ph, pdr, pa), (eh, ed, ea) = _poisson_elo_1x2(m, r.home_team_id, r.away_team_id, bool(r.neutral))
            cache.append((r.result, ph, pdr, pa, eh, ed, ea))
        for b in grid_blend:
            ll = acc = n = 0
            for res, ph, pdr, pa, eh, ed, ea in cache:
                H = b*ph+(1-b)*eh; D = b*pdr+(1-b)*ed; A = b*pa+(1-b)*ea
                s = H+D+A; H,D,A = H/s,D/s,A/s
                probs = {'H':H,'D':D,'A':A}
                ll -= np.log(max(probs[res], 1e-9))
                acc += (max(probs, key=probs.get) == res); n += 1
            out.append((xi, b, ll/n, acc/n, n))
    out.sort(key=lambda x: x[2])   # by log-loss
    print(f"\n  {'xi':>8} {'blend':>6} {'logloss':>9} {'acc':>7} {'n':>5}")
    for xi, b, ll, acc, n in out:
        tag = "  ← best" if (xi, b) == (out[0][0], out[0][1]) else ""
        print(f"  {xi:8.4f} {b:6.2f} {ll:9.4f} {acc:7.1%} {n:5d}{tag}")
    return out[0]   # best (xi, blend, ll, acc, n)


def resolve(model, name):
    """Resolve a team name to id (exact, then case-insensitive contains)."""
    n2i = model['name_to_id']
    if name in n2i: return n2i[name]
    low = name.lower()
    hits = [v for k, v in n2i.items() if low in k.lower() or k.lower() in low]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--predict-wc", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--home"); ap.add_argument("--away")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)

    if args.tune:
        df = load_intl(conn)
        print(f"  Loaded {len(df):,} clean senior international matches")
        tune(df)

    if args.train:
        df = load_intl(conn)
        print(f"  Loaded {len(df):,} completed international matches")
        model = fit(df)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        # quick top-10 Elo sanity print
        top = sorted(model['elo'].items(), key=lambda x: -x[1])[:12]
        print("\n  Top nations by Elo:")
        for tid, r in top:
            print(f"    {model['names'].get(tid,'?'):22s} {r:.0f}")
        print(f"\n✅ Saved → {MODEL_PATH}")

    if args.predict_wc:
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        up = pd.read_sql_query(
            "SELECT * FROM intl_fixtures WHERE league_id=1 AND season=2026 "
            "AND result IS NULL ORDER BY date", conn)
        print(f"\n{'='*78}\n  WORLD CUP 2026 — UPCOMING MATCH PREDICTIONS ({len(up)} fixtures)\n{'='*78}")
        for r in up.itertuples():
            pr = predict(model, r.home_team_id, r.away_team_id, neutral=True)
            bh, bd, ba = pr['blend_1x2']
            print(f"\n  {r.date}  {r.round}")
            print(f"  {r.home_team} vs {r.away_team}   (Elo {pr['elo_home']} v {pr['elo_away']})")
            print(f"    1X2 : {r.home_team[:14]} {bh:.0%} | Draw {bd:.0%} | {r.away_team[:14]} {ba:.0%}")
            print(f"    xG  : {pr['lambda_home']} - {pr['lambda_away']}   "
                  f"O2.5 {pr['over_2_5']:.0%} | BTTS {pr['btts_yes']:.0%}")
            print(f"    Likely scores: " + ", ".join(f"{s} ({p:.0%})" for s, p in pr['top_scores']))

    if args.predict:
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        hid, aid = resolve(model, args.home), resolve(model, args.away)
        if not hid or not aid:
            print(f"❌ Could not resolve: home={args.home}->{hid}, away={args.away}->{aid}"); sys.exit(1)
        pr = predict(model, hid, aid, neutral=args.neutral)
        print(json.dumps({**pr, 'home': model['names'][hid], 'away': model['names'][aid]}, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
