# Usage:
#   python src/finalize_winner.py                 # auto-picks best from cv_leaderboard.csv
#   python src/finalize_winner.py --tag K20_HA50_SR20_xg0_L101_rest1_cc0_lr0.06_it600_l20.0_calisotonic

import argparse, json, platform, os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import log_loss, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Reuse your builder + dataclasses from experiment_runner.py
from experimental import (
    ExperimentConfig, EloParams, FeatureFlags, ModelParams,
    build_feature_tables
)

ART_DIR = Path("artifacts")
RANDOM_STATE = 42

def load_cfg_from_tag(tag: str) -> ExperimentConfig:
    cfg_path = ART_DIR / f"v1_{tag}" / "config.json"
    d = json.loads(cfg_path.read_text())
    return ExperimentConfig(
        elo=EloParams(**d["elo"]),
        feats=FeatureFlags(**d["feats"]),
        model=ModelParams(**d["model"]),
    )

def choose_tag(args) -> str:
    if args.tag:
        return args.tag
    cv_path = ART_DIR / "cv_leaderboard.csv"
    if not cv_path.exists():
        raise SystemExit("cv_leaderboard.csv not found. Run src/cv_refit.py first or pass --tag.")
    cv = pd.read_csv(cv_path).sort_values("cv_mean_log_loss")
    return cv.iloc[0]["tag"]

def build_preprocessor(feature_cols):
    num = [c for c in feature_cols if c not in ("league_id","month")]
    cat = [c for c in ("league_id","month") if c in feature_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
        ],
        remainder="drop"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, help="Tag of the winning config to finalize")
    args = ap.parse_args()

    tag = choose_tag(args)
    print(f"Finalizing winner: {tag}")

    cfg = load_cfg_from_tag(tag)

    # Build features (same as experiments)
    train_df, valid_df, test_df, feature_cols, XY = build_feature_tables(cfg)

    # Split explicitly
    X_tr, y_tr = XY(train_df)                                    # 2015–2022
    # valid_df should already be 2023, but filter just in case:
    X_cal, y_cal = XY(valid_df[valid_df["season_year"] == 2023])
    X_te,  y_te  = XY(test_df)                                   # 2024

    # Model
    pre = build_preprocessor(feature_cols)
    base = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=cfg.model.learning_rate,
        max_iter=cfg.model.max_iter,
        l2_regularization=cfg.model.l2,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("prep", pre), ("clf", base)])
    pipe.fit(X_tr, y_tr)

    # Calibrate on 2023 (same method as your search)
    cal = CalibratedClassifierCV(pipe, method=cfg.model.calibration, cv="prefit")
    cal.fit(X_cal, y_cal)

    # Evaluate
    def score(name, X, y):
        p = cal.predict_proba(X)
        ll = log_loss(y, p, labels=[0,1,2])
        acc = accuracy_score(y, p.argmax(1))
        print(f"{name}: log_loss={ll:.4f}, acc={acc:.3f}")
        return ll, acc

    print("\n=== Final evaluation ===")
    _ = score("Calib (2023)", X_cal, y_cal)
    ll_test, acc_test = score("Test  (2024)", X_te, y_te)

    # Save artifacts
    out = ART_DIR / f"final_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    dump(cal, out / "model_calibrated.joblib")
    (out / "features.txt").write_text("\n".join(feature_cols))
    (out / "config.json").write_text(json.dumps({
        "elo": vars(cfg.elo), "feats": vars(cfg.feats), "model": vars(cfg.model)
    }, indent=2))

    report = {
        "train_years": "2015-2022",
        "calibration_year": 2023,
        "test_year": 2024,
        "scores": {"test_log_loss": float(ll_test), "test_acc": float(acc_test)},
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # Save test probabilities
    p = cal.predict_proba(X_te)
    pd.DataFrame({
        "fixture_id": test_df["fixture_id"].values,
        "season_year": test_df["season_year"].values,
        "league_id": test_df["league_id"].values,
        "p_home": p[:,0], "p_draw": p[:,1], "p_away": p[:,2]
    }).to_csv(out / "test_predictions.csv", index=False)

    print(f"\nSaved final model → {out/'model_calibrated.joblib'}")
    print(f"Artifacts folder → {out}")

if __name__ == "__main__":
    main()
