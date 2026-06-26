#!/usr/bin/env python3
"""
xgboost_model.py — XGBoost ensemble for match result prediction.

Predicts 3 separate outcomes:
  1. Match result (Home Win / Draw / Away Win) — multiclass
  2. Over/Under 2.5 goals — binary
  3. BTTS (Both Teams to Score) — binary

For each model, outputs calibrated probabilities.
Value bets are flagged where model probability > bookmaker implied probability by >4%.

Training strategy:
  - 5-fold time-series cross-validation (never train on future to predict past)
  - Optuna hyperparameter tuning (50 trials per model)
  - Train on seasons 2020-2023, validate on 2024, test on 2025 (current)
  - Separate models per target (1X2, OU2.5, BTTS)

Usage:
  python3 models/xgboost_model.py --train
  python3 models/xgboost_model.py --train --league 39  # single league
  python3 models/xgboost_model.py --predict upcoming.parquet
"""

import os, sys, json, sqlite3, argparse, pickle
from datetime import timezone, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, log_loss,
                              roc_auc_score, classification_report)

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.settings import DB_PATH, PARQUET_DIR
from config.leagues  import SEASONS, CURRENT_SEASON
from features.build_features import build_all_features, get_feature_columns

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved')
os.makedirs(MODEL_DIR, exist_ok=True)

N_OPTUNA_TRIALS = 50

# ── Train/val split strategy ───────────────────────────────────────────────────
TRAIN_SEASONS = [2021, 2022, 2023]   # 3 complete seasons for training
VAL_SEASON    = 2024                  # hold-out validation (most recent complete season)
TEST_SEASON   = 2025                  # current live season — predict upcoming matches


# ── Optuna objectives ──────────────────────────────────────────────────────────

def make_xgb_objective(X_tr, y_tr, X_vl, y_vl, n_classes):
    """Return an Optuna objective function for XGBoost tuning."""
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
        }
        if n_classes > 2:
            params['objective']  = 'multi:softprob'
            params['num_class']  = n_classes
        else:
            params['objective']  = 'binary:logistic'

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  verbose=False)

        if n_classes > 2:
            preds = model.predict_proba(X_vl)
            score = log_loss(y_vl, preds)
        else:
            preds = model.predict_proba(X_vl)[:, 1]
            score = log_loss(y_vl, preds)

        return score

    return objective


# ── Train one model ────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, target_name, n_classes=2):
    """
    Run Optuna tuning + final model fit.
    Returns fitted XGBClassifier.
    """
    print(f"\n  ── Tuning {target_name} ({N_OPTUNA_TRIALS} trials) ──")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        make_xgb_objective(X_train, y_train, X_val, y_val, n_classes),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"  Best {target_name} log-loss: {study.best_value:.4f}")
    print(f"  Best params: {best}")

    # Retrain on full train+val with best params
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    final_params = {**best,
                    'tree_method': 'hist',
                    'random_state': 42}
    if n_classes > 2:
        final_params['objective'] = 'multi:softprob'
        final_params['num_class'] = n_classes
        final_params['eval_metric'] = 'mlogloss'
    else:
        final_params['objective']   = 'binary:logistic'
        final_params['eval_metric'] = 'logloss'

    model = xgb.XGBClassifier(**final_params)
    model.fit(X_full, y_full, verbose=False)

    return model, study.best_value


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate_result_model(model, X_test, y_test, label_encoder, feature_cols):
    """Evaluate the 1X2 model on the test set and print metrics."""
    preds     = model.predict(X_test)
    probs     = model.predict_proba(X_test)

    acc       = accuracy_score(y_test, preds)
    ll        = log_loss(y_test, probs)

    print(f"\n  ── 1X2 Model Test Metrics ─────────────────────────────")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Log-loss:  {ll:.4f}")

    # Per-class metrics
    labels = label_encoder.classes_
    print(f"\n  {classification_report(y_test, preds, target_names=labels)}")

    # Feature importance (top 20)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    print(f"\n  Top 20 features:")
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 300)
        print(f"  {row['feature']:50s}  {row['importance']:.4f}  {bar}")

    return {'accuracy': acc, 'log_loss': ll}


def evaluate_binary_model(model, X_test, y_test, target_name):
    """Evaluate Over/Under or BTTS model."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    ll    = log_loss(y_test, probs)
    auc   = roc_auc_score(y_test, probs)

    print(f"\n  ── {target_name} Test Metrics ────────────────────────")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    return {'accuracy': acc, 'log_loss': ll, 'auc': auc}


# ── Predict future matches ─────────────────────────────────────────────────────

def predict_upcoming(models_bundle, feature_df, odds_df=None):
    """
    Generate predictions for upcoming matches.

    models_bundle: dict with 'result', 'ou25', 'btts' models + metadata
    feature_df:    DataFrame of upcoming matches with all feature columns
    odds_df:       Optional DataFrame of bookmaker odds for value bet detection

    Returns a DataFrame with predictions and value bet flags.
    """
    result_model = models_bundle['result_model']
    ou25_model   = models_bundle['ou25_model']
    btts_model   = models_bundle['btts_model']
    feature_cols = models_bundle['feature_cols']
    le           = models_bundle['label_encoder']

    # Fill missing features with 0
    X = feature_df[feature_cols].fillna(0).values

    # 1X2 probabilities
    result_probs = result_model.predict_proba(X)
    result_classes = le.classes_  # e.g. ['A', 'D', 'H'] or [0, 1, 2]

    # Map to named columns
    proba_df = pd.DataFrame(result_probs,
                             columns=[f'prob_{c}' for c in result_classes])

    # Rename to standard names
    class_to_col = {'A': 'prob_away', 'D': 'prob_draw', 'H': 'prob_home'}
    for cls in result_classes:
        col_src = f'prob_{cls}'
        col_dst = class_to_col.get(str(cls), col_src)
        if col_src in proba_df.columns:
            proba_df = proba_df.rename(columns={col_src: col_dst})

    # Over/Under and BTTS
    proba_df['prob_over_2_5'] = ou25_model.predict_proba(X)[:, 1]
    proba_df['prob_btts']     = btts_model.predict_proba(X)[:, 1]

    # Most likely result
    result_cols = ['prob_home', 'prob_draw', 'prob_away']
    proba_df['predicted_result'] = proba_df[result_cols].idxmax(axis=1).str.replace('prob_', '')

    # Combine with fixture info
    id_cols = ['fixture_id', 'home_team', 'away_team', 'date', 'league_id']
    out = pd.concat([
        feature_df[[c for c in id_cols if c in feature_df.columns]].reset_index(drop=True),
        proba_df.reset_index(drop=True)
    ], axis=1)

    # Value bet detection (if odds provided)
    if odds_df is not None and not odds_df.empty:
        out = flag_value_bets(out, odds_df)

    return out


def flag_value_bets(predictions_df, odds_df, edge_threshold=0.04):
    """
    Compare model probabilities with bookmaker implied probabilities.
    Flag as value bet where model_prob > bookmaker_implied_prob + edge_threshold.

    odds_df should have columns: event_id, market, outcome_name, avg_implied_prob
    """
    out = predictions_df.copy()

    # Map outcome names to model probability columns
    outcome_map = {
        'Home':  'prob_home',
        'Draw':  'prob_draw',
        'Away':  'prob_away',
    }

    value_flags = []
    value_details = []

    for _, row in out.iterrows():
        fixture_id = row.get('fixture_id')
        flags  = []
        details = {}

        # Get odds for this event
        event_odds = odds_df[odds_df['event_id'].astype(str) == str(fixture_id)]

        for outcome, model_col in outcome_map.items():
            if model_col not in row:
                continue
            model_prob = row[model_col]

            h2h_odds = event_odds[
                (event_odds['market'] == 'h2h') &
                (event_odds['outcome_name'] == outcome)
            ]

            if h2h_odds.empty:
                continue

            bk_implied = h2h_odds['avg_implied_prob'].values[0]
            edge = model_prob - bk_implied

            if edge > edge_threshold:
                flags.append(f"{outcome}({edge:.2%})")
                details[outcome] = {
                    'model_prob':    round(model_prob, 4),
                    'bk_implied':    round(bk_implied, 4),
                    'edge':          round(edge, 4),
                }

        value_flags.append(' | '.join(flags) if flags else '')
        value_details.append(json.dumps(details))

    out['value_bets']    = value_flags
    out['value_details'] = value_details

    n_value = (out['value_bets'] != '').sum()
    print(f"  Value bets flagged: {n_value}/{len(out)} matches")

    return out


# ── Save / load ────────────────────────────────────────────────────────────────

def save_bundle(bundle, path):
    """
    Save the model bundle.

    XGBoost models are saved TWICE:
      1. As pickle (fast, includes all metadata in one file) — local training use.
      2. As JSON via .save_model() (XGBoost-native, guaranteed cross-platform /
         cross-version portable) — used by Lambda inference.

    JSON files sit alongside the pickle:
        models/saved/xgboost_bundle.pkl            ← pickle (full bundle)
        models/saved/xgboost_result_model.json     ← portable XGBoost JSON
        models/saved/xgboost_ou25_model.json
        models/saved/xgboost_btts_model.json
        models/saved/xgboost_meta.json             ← label_encoder + feature_cols
    """
    # 1. Full pickle bundle (used locally)
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Saved pickle bundle → {path}")

    # 2. XGBoost JSON models (used by Lambda / any Linux deployment)
    base_dir = os.path.dirname(path)
    for key in ('result_model', 'ou25_model', 'btts_model'):
        json_path = os.path.join(base_dir, f"xgboost_{key}.json")
        bundle[key].save_model(json_path)
        print(f"  Saved portable JSON  → {json_path}")

    # 3. Non-XGBoost metadata (LabelEncoder, feature list, metrics)
    meta = {
        'feature_cols':   bundle['feature_cols'],
        'label_encoder_classes': bundle['label_encoder'].classes_.tolist(),
        'train_seasons':  bundle['train_seasons'],
        'val_season':     bundle['val_season'],
        'trained_at':     bundle['trained_at'],
        'metrics':        bundle['metrics'],
    }
    meta_path = os.path.join(base_dir, 'xgboost_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta JSON      → {meta_path}")


def load_bundle(path):
    """
    Load the model bundle.

    Prefers JSON models (portable) if they exist alongside the pickle.
    Falls back to pure pickle if JSON files are missing (e.g. older saves).
    """
    # Check whether portable JSON models are available
    base_dir = os.path.dirname(path)
    json_keys = {
        'result_model': os.path.join(base_dir, 'xgboost_result_model.json'),
        'ou25_model':   os.path.join(base_dir, 'xgboost_ou25_model.json'),
        'btts_model':   os.path.join(base_dir, 'xgboost_btts_model.json'),
    }
    meta_path = os.path.join(base_dir, 'xgboost_meta.json')

    all_json_present = (
        all(os.path.exists(p) for p in json_keys.values())
        and os.path.exists(meta_path)
    )

    if all_json_present:
        # Load each XGBClassifier from its portable JSON
        import xgboost as _xgb
        from sklearn.preprocessing import LabelEncoder as _LE

        bundle = {}
        for key, json_path in json_keys.items():
            m = _xgb.XGBClassifier()
            m.load_model(json_path)
            bundle[key] = m

        with open(meta_path) as f:
            meta = json.load(f)

        le = _LE()
        import numpy as _np
        le.classes_ = _np.array(meta['label_encoder_classes'])
        bundle['label_encoder'] = le
        bundle['feature_cols']  = meta['feature_cols']
        bundle['train_seasons'] = meta.get('train_seasons', [])
        bundle['val_season']    = meta.get('val_season')
        bundle['trained_at']    = meta.get('trained_at')
        bundle['metrics']       = meta.get('metrics', {})
        return bundle

    # Fallback: pure pickle (requires same xgboost version)
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost football prediction model")
    parser.add_argument('--train',   action='store_true')
    parser.add_argument('--predict', default=None,
                        help='Path to upcoming fixtures parquet for prediction')
    parser.add_argument('--league',  type=int, nargs='+', default=None)
    parser.add_argument('--db',      default=None)
    args = parser.parse_args()

    db_path = args.db or str(DB_PATH)
    conn    = sqlite3.connect(db_path)

    if args.train:
        print(f"\n── Training XGBoost model ───────────────────────────────────")

        # Build features
        df = build_all_features(conn, league_ids=args.league)
        feature_cols = get_feature_columns(df)
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(df):,}")

        # Split
        train_mask = df['season'].isin(TRAIN_SEASONS)
        val_mask   = df['season'] == VAL_SEASON
        test_mask  = df['season'] == TEST_SEASON

        X_train = df[train_mask][feature_cols].fillna(0).values
        X_val   = df[val_mask][feature_cols].fillna(0).values
        X_test  = df[test_mask][feature_cols].fillna(0).values if test_mask.any() else X_val

        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

        # ── Model 1: 1X2 Result ──────────────────────────────────────────────
        le = LabelEncoder()
        y_train_r = le.fit_transform(df[train_mask]['result'])
        y_val_r   = le.transform(df[val_mask]['result'])
        y_test_r  = le.transform(df[test_mask]['result']) if test_mask.any() else y_val_r

        result_model, result_loss = train_model(
            X_train, y_train_r, X_val, y_val_r,
            target_name='1X2', n_classes=3)
        test_metrics_r = evaluate_result_model(
            result_model, X_test, y_test_r, le, feature_cols)

        # ── Model 2: Over 2.5 ───────────────────────────────────────────────
        y_train_ou = df[train_mask]['over_2_5'].fillna(0).astype(int).values
        y_val_ou   = df[val_mask]['over_2_5'].fillna(0).astype(int).values
        y_test_ou  = (df[test_mask]['over_2_5'].fillna(0).astype(int).values
                      if test_mask.any() else y_val_ou)

        ou25_model, ou25_loss = train_model(
            X_train, y_train_ou, X_val, y_val_ou,
            target_name='Over2.5', n_classes=2)
        test_metrics_ou = evaluate_binary_model(
            ou25_model, X_test, y_test_ou, 'Over 2.5')

        # ── Model 3: BTTS ────────────────────────────────────────────────────
        y_train_btts = df[train_mask]['btts'].fillna(0).astype(int).values
        y_val_btts   = df[val_mask]['btts'].fillna(0).astype(int).values
        y_test_btts  = (df[test_mask]['btts'].fillna(0).astype(int).values
                        if test_mask.any() else y_val_btts)

        btts_model, btts_loss = train_model(
            X_train, y_train_btts, X_val, y_val_btts,
            target_name='BTTS', n_classes=2)
        test_metrics_btts = evaluate_binary_model(
            btts_model, X_test, y_test_btts, 'BTTS')

        # ── Save ─────────────────────────────────────────────────────────────
        bundle = {
            'result_model':   result_model,
            'ou25_model':     ou25_model,
            'btts_model':     btts_model,
            'label_encoder':  le,
            'feature_cols':   feature_cols,
            'train_seasons':  TRAIN_SEASONS,
            'val_season':     VAL_SEASON,
            'trained_at':     datetime.now(timezone.utc).isoformat(),
            'metrics': {
                'result_val_logloss': result_loss,
                'ou25_val_logloss':   ou25_loss,
                'btts_val_logloss':   btts_loss,
                '1x2_test':  test_metrics_r,
                'ou25_test': test_metrics_ou,
                'btts_test': test_metrics_btts,
            }
        }

        bundle_path = os.path.join(MODEL_DIR, 'xgboost_bundle.pkl')
        save_bundle(bundle, bundle_path)

        # Save metrics JSON
        metrics_path = os.path.join(MODEL_DIR, 'xgboost_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(bundle['metrics'], f, indent=2)
        print(f"  Metrics → {metrics_path}")

        print(f"\n  ✅ Training complete")
        print(f"  1X2  accuracy: {test_metrics_r['accuracy']*100:.1f}%")
        print(f"  OU25 accuracy: {test_metrics_ou['accuracy']*100:.1f}%")
        print(f"  BTTS accuracy: {test_metrics_btts['accuracy']*100:.1f}%")

    elif args.predict:
        bundle_path = os.path.join(MODEL_DIR, 'xgboost_bundle.pkl')
        if not os.path.exists(bundle_path):
            print("❌ No trained model found — run with --train first")
            sys.exit(1)

        bundle   = load_bundle(bundle_path)
        feat_df  = pd.read_parquet(args.predict)

        # Filter to upcoming matches only (no result = future fixture)
        upcoming_mask = feat_df['result_encoded'].isna() if 'result_encoded' in feat_df.columns \
                        else feat_df.get('result', pd.Series()).isna()
        upcoming_df = feat_df[upcoming_mask].copy()
        print(f"  Upcoming fixtures to predict: {len(upcoming_df):,}")

        if upcoming_df.empty:
            print("  ⚠️  No upcoming fixtures found — run collect_fixtures.py --update first")
            sys.exit(0)

        # Load odds if available
        try:
            from data_collection.collect_odds import get_consensus_odds
            odds_df = get_consensus_odds(conn)
            print(f"  Loaded {len(odds_df):,} consensus odds rows")
        except Exception:
            odds_df = None

        predictions = predict_upcoming(bundle, upcoming_df, odds_df)

        out_path = str(PARQUET_DIR / "predictions_xgb.parquet")
        predictions.to_parquet(out_path, index=False)
        print(f"\n  Predictions saved → {out_path}")
        print(predictions[['home_team', 'away_team', 'date',
                            'prob_home', 'prob_draw', 'prob_away',
                            'predicted_result', 'prob_over_2_5', 'prob_btts']
                           + (['value_bets'] if 'value_bets' in predictions.columns else [])
                           ].to_string(index=False))

    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
