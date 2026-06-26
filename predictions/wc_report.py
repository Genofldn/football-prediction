#!/usr/bin/env python3
"""
wc_report.py — Thorough World Cup prediction report with value-bet detection.

Per fixture: 1X2, Over/Under 2.5, BTTS, top scorelines (model) + bookmaker
consensus odds and value flags (model prob exceeds de-vigged implied prob by >4%).
BTTS shows model only (bookmakers don't offer BTTS on the WC).

Usage:
  python3 predictions/wc_report.py
"""
import os, sys, pickle, sqlite3, re
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.national_team_model import predict

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.db")
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "national_team_model.pkl")
VALUE_EDGE = 0.04   # flag when model prob exceeds de-vigged market prob by > 4 percentage points

def norm_tokens(name):
    return frozenset(re.sub(r"[^a-z ]", "", (name or "").lower()).split())

def teams_match(a, b):
    ta, tb = norm_tokens(a), norm_tokens(b)
    return ta == tb or ta <= tb or tb <= ta

def load_odds(conn):
    """Return {event_key: {'h2h': {team_name:(devig,best)}, 'draw':(devig,best), 'ou':{'Over':(devig,best),'Under':(...)}}}"""
    rows = conn.execute(
        "SELECT event_id, home_team, away_team, market, outcome_name, outcome_price "
        "FROM odds WHERE sport_key='soccer_fifa_world_cup' AND outcome_price>0").fetchall()
    ev = {}
    for eid, h, a, mkt, name, price in rows:
        e = ev.setdefault(eid, {"home": h, "away": a, "h2h": {}, "ou": {}})
        if mkt == "h2h":
            e["h2h"].setdefault(name, []).append(price)
        elif mkt == "totals" and name in ("Over 2.5", "Under 2.5"):
            e["ou"].setdefault(name.split()[0], []).append(price)
    # collapse to (devigged implied, best price)
    out = {}
    for eid, e in ev.items():
        h2h = {k: (np.mean([1/p for p in v]), max(v)) for k, v in e["h2h"].items()}
        s = sum(d for d, _ in h2h.values())
        h2h = {k: (d/s, b) for k, (d, b) in h2h.items()} if s else h2h
        ou = {k: (np.mean([1/p for p in v]), max(v)) for k, v in e["ou"].items()}
        so = sum(d for d, _ in ou.values())
        ou = {k: (d/so, b) for k, (d, b) in ou.items()} if so else ou
        out[eid] = {"home": e["home"], "away": e["away"], "h2h": h2h, "ou": ou}
    return out

def find_odds(odds, home, away):
    for e in odds.values():
        if (teams_match(e["home"], home) and teams_match(e["away"], away)) or \
           (teams_match(e["home"], away) and teams_match(e["away"], home)):
            return e
    return None

def odds_for_team(h2h, team):
    for name, val in h2h.items():
        if name != "Draw" and teams_match(name, team):
            return val
    return None

def main():
    conn = sqlite3.connect(DB)
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    odds = load_odds(conn)
    up = conn.execute(
        "SELECT date, round, home_team, home_team_id, away_team, away_team_id "
        "FROM intl_fixtures WHERE league_id=1 AND season=2026 AND result IS NULL ORDER BY date").fetchall()

    L = []
    val_bets = []
    L.append("="*82)
    L.append("  FIFA WORLD CUP 2026 — MATCH PREDICTIONS & VALUE BETS")
    L.append(f"  Generated {datetime.now():%Y-%m-%d %H:%M} | model: national-team Elo + Dixon-Coles Poisson")
    L.append(f"  Value flag = model probability exceeds de-vigged market probability by >{VALUE_EDGE:.0%}")
    L.append("="*82)

    for date, rnd, h, hid, a, aid in up:
        pr = predict(model, hid, aid, neutral=True)
        ph, pdr, pa = pr['blend_1x2']
        e = find_odds(odds, h, a)
        L.append(f"\n{date}  {rnd}")
        L.append(f"  {h}  vs  {a}      (Elo {pr['elo_home']} v {pr['elo_away']})")
        L.append(f"  Expected goals: {pr['lambda_home']} - {pr['lambda_away']}")

        # ── line printer. flag_value=True only for markets where the model is validated (1X2) ──
        def vline(label, mprob, mkt, flag_value):
            nonlocal val_bets
            if mkt:
                devig, best = mkt
                edge = mprob - devig
                flag = ""
                if flag_value and edge > VALUE_EDGE:
                    ev = mprob*best - 1
                    flag = f"   ★ VALUE  (edge +{edge:.0%}, odds {best:.2f}, EV +{ev:.0%})"
                    val_bets.append(f"{date}  {h} v {a}: {label}  model {mprob:.0%} vs mkt {devig:.0%}  @ {best:.2f}  (EV +{ev:.0%})")
                return f"    {label:22s} model {mprob:4.0%} | mkt {devig:4.0%} | best {best:5.2f}{flag}"
            return f"    {label:22s} model {mprob:4.0%} | (no odds)"

        L.append("  1X2:  [value-checked]")
        L.append(vline(f"{h} win", ph, odds_for_team(e["h2h"], h) if e else None, True))
        L.append(vline("Draw", pdr, e["h2h"].get("Draw") if e else None, True))
        L.append(vline(f"{a} win", pa, odds_for_team(e["h2h"], a) if e else None, True))

        # ── Over/Under 2.5 — informational only (model runs ~10% under market on this slate) ──
        L.append("  Goals (informational — not value-checked, see notes):")
        L.append(vline("Over 2.5",  pr['over_2_5'],  e["ou"].get("Over")  if e else None, False))
        L.append(vline("Under 2.5", pr['under_2_5'], e["ou"].get("Under") if e else None, False))

        # ── BTTS (model only) ──
        L.append(f"  BTTS:   Yes model {pr['btts_yes']:.0%} | No model {pr['btts_no']:.0%}   (no market offered)")

        # ── Scorelines ──
        L.append("  Most likely scorelines: " + ",  ".join(f"{s} ({p:.0%})" for s, p in pr['top_scores']))

    # ── Value summary ──
    L.append("\n" + "="*82)
    if val_bets:
        L.append(f"  ★ 1X2 VALUE BETS ({len(val_bets)}):")
        for v in val_bets:
            L.append(f"    {v}")
    else:
        L.append("  No 1X2 value bets cleared the threshold.")
    L.append("="*82)

    # ── Methodology & calibration notes (honesty) ──
    L.append("\n  MODEL NOTES — read before betting:")
    L.append("  • Model: national-team Elo + Dixon-Coles Poisson, fit on 5,740 senior")
    L.append("    internationals (2018-2026). Time-decay xi=0.0010, Poisson-primary.")
    L.append("  • 1X2 backtest (554 held-out competitive matches): 65% accuracy, 0.776 log-loss,")
    L.append("    draw calibration model 22.6% vs actual 21.7% (+1.0%). 1X2 value flags shown")
    L.append("    only here, and are statistically defensible.")
    L.append("  • CAVEAT on group-stage flags: final group games often see qualified teams rotate")
    L.append("    squads — market prices this, model cannot (no lineup data). Treat dead-rubber")
    L.append("    draw/underdog value with caution; knockout (R32) flags are cleaner.")
    L.append("  • O/U 2.5 is INFORMATIONAL ONLY. The model runs ~10% below the market on Over 2.5")
    L.append("    across this slate (model avg 40% vs market 50%). Backtest bias is only -2.7%,")
    L.append("    so this gap is mostly model conservatism on tournament goals, NOT genuine value.")
    L.append("    Do NOT bet the Under signals on that basis.")
    L.append("  • BTTS: model view only — bookmakers do not offer BTTS on the World Cup.")
    L.append("  • All matches treated as neutral venue. Friendlies down-weighted vs competitive.")
    L.append("  • Value = model prob exceeds de-vigged market prob by >4%. EV uses best price across")
    L.append("    41 (1X2) bookmakers. Underdog/draw edges partly reflect favourite-longshot bias.")
    L.append("="*82)

    report = "\n".join(L)
    print(report)
    out_path = os.path.join(os.path.dirname(__file__), "reports", f"wc_report_{datetime.now():%Y%m%d}.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved → {out_path}")
    conn.close()

if __name__ == "__main__":
    main()
