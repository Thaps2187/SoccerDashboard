# Time-series CV for the top-K configs from your sweep
import json, os
import pandas as pd
from pathlib import Path
from joblib import load
from experimental import (  # reuse your builders
    ExperimentConfig, EloParams, FeatureFlags, ModelParams,
    build_feature_tables
)
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

K_SHORTLIST = 20
FOLDS = [
    (2018, 2019),
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
]

def load_cfg_from_folder(run_folder: Path) -> ExperimentConfig:
    cfg = json.loads((run_folder / "config.json").read_text())
    return ExperimentConfig(
        elo=EloParams(**cfg["elo"]),
        feats=FeatureFlags(**cfg["feats"]),
        model=ModelParams(**cfg["model"]),
    )

def score_config_cv(run_folder: Path):
    cfg = load_cfg_from_folder(run_folder)
    train_df, valid_df, test_df, feature_cols, XY = build_feature_tables(cfg)

    # Preprocessor (same as runner)
    numeric_features = [c for c in feature_cols if c not in ["league_id","month"]]
    categorical_features = [c for c in ["league_id","month"] if c in feature_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop"
    )

    scores = []
    for train_end, valid_year in FOLDS:
        fold_train = train_df[train_df["season_year"] <= train_end]
        fold_valid = (pd.concat([train_df, valid_df, test_df])  # has all years
                      [lambda d: d["season_year"] == valid_year])

        X_tr, y_tr = XY(fold_train)
        X_va, y_va = XY(fold_valid)

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=cfg.model.learning_rate,
            max_iter=cfg.model.max_iter,
            l2_regularization=cfg.model.l2,
            random_state=42
        )
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        p = pipe.predict_proba(X_va)
        scores.append(log_loss(y_va, p, labels=[0,1,2]))

    return float(pd.Series(scores).mean()), float(pd.Series(scores).std())

def main():
    artifacts = Path("artifacts")
    lb = pd.read_csv(artifacts / "leaderboard.csv").sort_values("valid_log_loss").head(K_SHORTLIST)
    rows = []
    for tag in lb["tag"].tolist():
        folder = artifacts / f"v1_{tag}"
        mean_ll, std_ll = score_config_cv(folder)
        rows.append({"tag": tag, "cv_mean_log_loss": mean_ll, "cv_std": std_ll})

    cv_table = pd.DataFrame(rows).sort_values("cv_mean_log_loss").reset_index(drop=True)
    cv_table.to_csv(artifacts / "cv_leaderboard.csv", index=False)
    print(cv_table.head(10))

if __name__ == "__main__":
    main()
