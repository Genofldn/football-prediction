#!/usr/bin/env python3
"""
generate_report.py — Combine XGBoost + Poisson predictions into a match report.

Merges predictions from both models, highlights value bets, and produces:
  - Console summary
  - HTML report (for email)
  - JSON file (for API/Lambda integration)

Usage:
  python3 predictions/generate_report.py
  python3 predictions/generate_report.py --email   # also send email via SNS/SES
"""

import os, sys, json
from datetime import timezone, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from config.settings import PARQUET_DIR
from config.leagues  import LEAGUE_BY_ID


PREDICTIONS_DIR = os.path.dirname(__file__)
REPORTS_DIR     = os.path.join(PREDICTIONS_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_predictions():
    """Load XGBoost and Poisson prediction parquets."""
    xgb_path     = str(PARQUET_DIR / "predictions_xgb.parquet")
    poisson_path = str(PARQUET_DIR / "predictions_poisson.parquet")

    xgb_df = pd.read_parquet(xgb_path) if os.path.exists(xgb_path) else pd.DataFrame()
    poi_df = pd.read_parquet(poisson_path) if os.path.exists(poisson_path) else pd.DataFrame()

    return xgb_df, poi_df


def merge_predictions(xgb_df, poi_df):
    """
    Merge XGBoost and Poisson predictions on fixture_id.
    Average the 1X2 probabilities from both models.
    """
    if xgb_df.empty and poi_df.empty:
        return pd.DataFrame()

    if xgb_df.empty:
        return poi_df

    if poi_df.empty:
        return xgb_df

    # Rename Poisson columns to distinguish
    poi_renamed = poi_df.rename(columns={
        'prob_home': 'poi_prob_home',
        'prob_draw': 'poi_prob_draw',
        'prob_away': 'poi_prob_away',
        'expected_home_goals': 'xg_home',
        'expected_away_goals': 'xg_away',
        'most_likely_scoreline': 'most_likely_scoreline',
    })

    merged = xgb_df.merge(
        poi_renamed[['fixture_id', 'poi_prob_home', 'poi_prob_draw', 'poi_prob_away',
                     'xg_home', 'xg_away', 'most_likely_scoreline']],
        on='fixture_id', how='left')

    # Ensemble: average where Poisson is available
    for outcome in ['home', 'draw', 'away']:
        xgb_col = f'prob_{outcome}'
        poi_col = f'poi_prob_{outcome}'
        ens_col = f'ens_prob_{outcome}'
        if poi_col in merged.columns:
            merged[ens_col] = merged[[xgb_col, poi_col]].mean(axis=1).round(4)
        else:
            merged[ens_col] = merged[xgb_col]

    # Ensemble predicted result
    ens_cols = ['ens_prob_home', 'ens_prob_draw', 'ens_prob_away']
    merged['ens_predicted_result'] = merged[ens_cols].idxmax(axis=1).str.replace('ens_prob_', '')

    return merged


def format_match_report(df):
    """Format predictions as a readable console/text report."""
    if df.empty:
        return "No upcoming fixtures found."

    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')

    lines = []
    lines.append(f"{'═'*70}")
    lines.append(f"  FOOTBALL PREDICTIONS — {datetime.now().strftime('%d %B %Y')}")
    lines.append(f"{'═'*70}")

    # Group by date
    current_date = None
    for _, row in df.iterrows():
        match_date = row.get('date')
        if pd.notnull(match_date):
            date_str = pd.Timestamp(match_date).strftime('%A %d %B')
        else:
            date_str = 'Date unknown'

        if date_str != current_date:
            current_date = date_str
            league_name = LEAGUE_BY_ID.get(row.get('league_id', 0), {}).get('name', '')
            lines.append(f"\n  ── {date_str} ──")

        home = row.get('home_team', '?')
        away = row.get('away_team', '?')

        ph = row.get('ens_prob_home', row.get('prob_home', 0))
        pd_ = row.get('ens_prob_draw', row.get('prob_draw', 0))
        pa = row.get('ens_prob_away', row.get('prob_away', 0))
        pred = row.get('ens_predicted_result', row.get('predicted_result', '?'))

        ou25  = row.get('prob_over_2_5', None)
        btts  = row.get('prob_btts', None)
        xg_h  = row.get('xg_home', None)
        xg_a  = row.get('xg_away', None)
        score = row.get('most_likely_scoreline', None)
        vbets = row.get('value_bets', '')

        # Main line
        pred_label = {'home': 'HOME WIN', 'draw': 'DRAW', 'away': 'AWAY WIN'}.get(pred, pred.upper())
        lines.append(
            f"\n  {home:28s}  vs  {away:28s}")
        lines.append(
            f"  H:{ph:.0%}  D:{pd_:.0%}  A:{pa:.0%}   → {pred_label}")

        # Goals / xG line
        extras = []
        if ou25 is not None:
            extras.append(f"Over2.5:{ou25:.0%}")
        if btts is not None:
            extras.append(f"BTTS:{btts:.0%}")
        if xg_h is not None and xg_a is not None:
            extras.append(f"xG:{xg_h:.2f}-{xg_a:.2f}")
        if score:
            extras.append(f"Most likely:{score}")
        if extras:
            lines.append(f"  {' | '.join(extras)}")

        # Value bets
        if vbets:
            lines.append(f"  ⚡ VALUE BET: {vbets}")

    lines.append(f"\n{'═'*70}")

    # Summary of value bets
    if 'value_bets' in df.columns:
        value_matches = df[df['value_bets'] != '']
        if not value_matches.empty:
            lines.append(f"\n  ── VALUE BETS SUMMARY ─────────────────────────────────")
            for _, row in value_matches.iterrows():
                home = row.get('home_team', '?')
                away = row.get('away_team', '?')
                vb   = row.get('value_bets', '')
                lines.append(f"  {home} vs {away}: {vb}")

    return '\n'.join(lines)


def generate_json_report(df):
    """Generate structured JSON output for Lambda/API integration."""
    if df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        rec = {
            'fixture_id':        int(row.get('fixture_id', 0)) if pd.notna(row.get('fixture_id')) else None,
            'home_team':         row.get('home_team'),
            'away_team':         row.get('away_team'),
            'date':              str(row.get('date', ''))[:10],
            'league_id':         int(row.get('league_id', 0)) if pd.notna(row.get('league_id')) else None,
            'predictions': {
                'prob_home':     round(float(row.get('ens_prob_home', row.get('prob_home', 0))), 4),
                'prob_draw':     round(float(row.get('ens_prob_draw', row.get('prob_draw', 0))), 4),
                'prob_away':     round(float(row.get('ens_prob_away', row.get('prob_away', 0))), 4),
                'predicted_result': row.get('ens_predicted_result', row.get('predicted_result')),
                'prob_over_2_5': round(float(row.get('prob_over_2_5', 0)), 4) if pd.notna(row.get('prob_over_2_5')) else None,
                'prob_btts':     round(float(row.get('prob_btts', 0)), 4) if pd.notna(row.get('prob_btts')) else None,
                'xg_home':       round(float(row.get('xg_home', 0)), 3) if pd.notna(row.get('xg_home')) else None,
                'xg_away':       round(float(row.get('xg_away', 0)), 3) if pd.notna(row.get('xg_away')) else None,
                'most_likely_scoreline': row.get('most_likely_scoreline'),
            },
            'value_bets':        row.get('value_bets', ''),
            'value_details':     json.loads(row.get('value_details', '{}') or '{}'),
            'generated_at':      datetime.now(timezone.utc).isoformat(),
        }
        records.append(rec)

    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate predictions report")
    parser.add_argument('--email', action='store_true',
                        help='Send report via email (requires AWS SNS setup)')
    args = parser.parse_args()

    print(f"\n── Generating predictions report ────────────────────────────")

    xgb_df, poi_df = load_predictions()
    print(f"  XGBoost predictions: {len(xgb_df)} matches")
    print(f"  Poisson predictions: {len(poi_df)} matches")

    merged = merge_predictions(xgb_df, poi_df)
    print(f"  Merged: {len(merged)} matches")

    if merged.empty:
        print("  ⚠️  No predictions to report — run XGBoost and Poisson predictions first")
        sys.exit(0)

    # Text report
    report_text = format_match_report(merged)
    print(report_text)

    # Save text report
    date_str     = datetime.now().strftime('%Y%m%d')
    text_path    = os.path.join(REPORTS_DIR, f"report_{date_str}.txt")
    with open(text_path, 'w') as f:
        f.write(report_text)
    print(f"\n  Text report → {text_path}")

    # JSON report
    json_records = generate_json_report(merged)
    json_path    = os.path.join(REPORTS_DIR, f"report_{date_str}.json")
    with open(json_path, 'w') as f:
        json.dump(json_records, f, indent=2)
    print(f"  JSON report → {json_path}")

    # Email via SNS (when Lambda integration is set up)
    if args.email:
        try:
            import boto3
            sns = boto3.client('sns', region_name='eu-west-2')
            # SNS topic ARN from Bitcoin pipeline
            topic_arn = "arn:aws:sns:eu-west-2:654654488711:bitcoin-prediction-alerts"
            sns.publish(
                TopicArn=topic_arn,
                Subject=f"Football Predictions — {datetime.now().strftime('%d %b %Y')}",
                Message=report_text,
            )
            print(f"  ✅ Email sent via SNS")
        except Exception as e:
            print(f"  ⚠️  Email failed: {e}")

    # Value bets count
    if 'value_bets' in merged.columns:
        n_value = (merged['value_bets'] != '').sum()
        print(f"\n  ⚡ Value bets flagged: {n_value}/{len(merged)}")

    print(f"\n  ✅ Report complete")


if __name__ == "__main__":
    main()
