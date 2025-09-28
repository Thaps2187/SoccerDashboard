# =========================
# Soccer v1 (wide match_stats)
# Features + Elo + Model
# =========================
# pip install pandas numpy sqlalchemy scikit-learn

from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------
# CONFIG — EDIT THESE
# -------------------------
DB_PATH = "/Users/thapeloclement/SoccerDashboard/soccer.db"
TRAIN_YEARS = list(range(2015, 2023))  # 2015–2022
VALID_YEAR = 2023
TEST_YEAR  = 2024

# Elo
ELO_START = 1500.0
ELO_HOME_ADV = 60.0
ELO_K = 16.0
ELO_SEASON_REGRESS = 0.30   # 30% toward league mean at season start

# If you have a 'matchday_text' like "MD4", set the column name; otherwise set to None.
MATCHDAY_TEXT_COL = "match_day"

# -------------------------
# Helpers
# -------------------------
def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def table_exists(conn, name: str) -> bool:
    return pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                       conn, params=[name]).shape[0] > 0

# -------------------------
# Load data
# -------------------------
conn = sqlite3.connect(DB_PATH)
if not table_exists(conn, "matches"):
    raise RuntimeError("Table 'matches' not found in DB_PATH.")

matches = pd.read_sql(f"""
    SELECT fixture_id, season_year, league_id, match_date,
           home_team_id, away_team_id, home_goals, away_goals
           {", " + MATCHDAY_TEXT_COL if MATCHDAY_TEXT_COL else ""}
    FROM matches
""", conn)
matches["match_date"] = to_dt(matches["match_date"])
matches = matches.sort_values(["league_id","match_date","fixture_id"]).reset_index(drop=True)

if MATCHDAY_TEXT_COL and MATCHDAY_TEXT_COL in matches.columns:
    matches["round_number"] = (
        matches[MATCHDAY_TEXT_COL].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    )

# Outcome label
def outcome(row):
    if pd.isna(row.home_goals) or pd.isna(row.away_goals): return np.nan
    if row.home_goals > row.away_goals: return "H"
    if row.home_goals < row.away_goals: return "A"
    return "D"
matches["outcome"] = matches.apply(outcome, axis=1)

# -------------------------
# Build team_match (two rows per fixture)
# -------------------------
home_tm = matches[["fixture_id","season_year","league_id","match_date",
                   "home_team_id","away_team_id","home_goals","away_goals"]].copy()
home_tm["team_side"] = "home"
home_tm["team_id"] = home_tm["home_team_id"]
home_tm["opp_id"] = home_tm["away_team_id"]
home_tm["gf"] = home_tm["home_goals"]
home_tm["ga"] = home_tm["away_goals"]

away_tm = matches[["fixture_id","season_year","league_id","match_date",
                   "home_team_id","away_team_id","home_goals","away_goals"]].copy()
away_tm["team_side"] = "away"
away_tm["team_id"] = away_tm["away_team_id"]
away_tm["opp_id"] = away_tm["home_team_id"]
away_tm["gf"] = away_tm["away_goals"]
away_tm["ga"] = away_tm["home_goals"]

team_match = pd.concat([home_tm, away_tm], ignore_index=True)
team_match["match_date"] = to_dt(team_match["match_date"])

# -------------------------
# Attach WIDE match_stats (map home_* / away_* to per-team cols)
# -------------------------
if table_exists(conn, "match_stats"):
    ms = pd.read_sql("""
        SELECT fixture_id,
               home_xgoals, away_xgoals,
               home_corners, away_corners,
               home_yellow_cards, away_yellow_cards,
               home_red_cards, away_red_cards,
               home_shots, away_shots,
               home_shots_on_goal, away_shots_on_goal,
               home_gk_saves, away_gk_saves,
               home_possession, away_possession,
               home_pass_accuracy, away_pass_accuracy,
               home_offsides, away_offsides,
               home_fouls, away_fouls
        FROM match_stats
    """, conn)
    team_match = team_match.merge(ms, on="fixture_id", how="left")

    # pick per-team value based on side
    side_is_home = team_match["team_side"].eq("home")

    def pick(home_col, away_col):
        return np.where(side_is_home, team_match[home_col], team_match[away_col])

    # Core features the model uses (names aligned with the rest of the script)
    team_match["shots"] = pick("home_shots", "away_shots")
    team_match["shots_on_target"] = pick("home_shots_on_goal", "away_shots_on_goal")
    team_match["corners"] = pick("home_corners", "away_corners")
    team_match["yellow_cards"] = pick("home_yellow_cards", "away_yellow_cards")
    team_match["red_cards"] = pick("home_red_cards", "away_red_cards")
    # optional extras (rolled but currently not in minimal feature list)
    team_match["xg"] = pick("home_xgoals", "away_xgoals")
    team_match["gk_saves"] = pick("home_gk_saves", "away_gk_saves")
    team_match["possession"] = pick("home_possession", "away_possession")
    team_match["pass_accuracy"] = pick("home_pass_accuracy", "away_pass_accuracy")
    team_match["offsides"] = pick("home_offsides", "away_offsides")
    team_match["fouls"] = pick("home_fouls", "away_fouls")
else:
    # create empty columns so the pipeline still runs
    for c in ["shots","shots_on_target","corners","yellow_cards","red_cards","xg","gk_saves",
              "possession","pass_accuracy","offsides","fouls"]:
        team_match[c] = np.nan

# columns that should be numeric
num_like_cols = [
    "shots","shots_on_target","corners","yellow_cards","red_cards",
    "xg","gk_saves","offsides","fouls","possession","pass_accuracy"
]

# 1) strip '%' for percent-ish columns, then convert to float
for c in ["possession", "pass_accuracy"]:
    if c in team_match.columns:
        team_match[c] = (
            team_match[c]
            .astype(str)
            .str.replace('%', '', regex=False)  # "67%" -> "67"
        )
        team_match[c] = pd.to_numeric(team_match[c], errors="coerce") / 100.0  # 0.67 scale

# 2) coerce all other numeric-like columns (handles strings like "12")
for c in ["shots","shots_on_target","corners","yellow_cards","red_cards","xg","gk_saves","offsides","fouls"]:
    if c in team_match.columns:
        team_match[c] = pd.to_numeric(team_match[c], errors="coerce")

# -------------------------
# Leak-safe rolling features (L5, L10) + rest days
# -------------------------
def add_rollings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("match_date").copy()

    roll_cols_sum = ["gf","ga","shots","shots_on_target","corners","yellow_cards","red_cards","xg","gk_saves","offsides","fouls"]
    # shift so current match is excluded
    for col in roll_cols_sum:
        df[f"{col}_prev"] = df[col].shift(1)

    # rolling sums (sensible for counts/xg)
    for w in [5, 10]:
        for col in ["gf","ga","shots","shots_on_target","corners","yellow_cards","red_cards","xg","gk_saves","offsides","fouls"]:
            df[f"{col}_L{w}"] = df[f"{col}_prev"].rolling(w, min_periods=1).sum()

    # rolling means for rates (possession/pass_accuracy)
    for col in ["possession","pass_accuracy"]:
        df[f"{col}_prev"] = df[col].shift(1)
        for w in [5, 10]:
            df[f"{col}_L{w}"] = df[f"{col}_prev"].rolling(w, min_periods=1).mean()

    # rest days
    df["rest_days"] = (df["match_date"] - df["match_date"].shift(1)).dt.days
    return df

team_match = (
    team_match
    .sort_values(["league_id","team_id","match_date"])
    .groupby(["league_id","team_id"], group_keys=False)
    .apply(add_rollings)
    .reset_index(drop=True)
)

# -------------------------
# Elo (pre-match)
# -------------------------
@dataclass
class EloState: rating: float

def exp_score(ra, rb, home_is_a: bool) -> float:
    adv = ELO_HOME_ADV if home_is_a else 0.0
    return 1.0 / (1.0 + 10 ** ((rb - (ra + adv))/400.0))

def update_elo(ra, rb, s_home) -> Tuple[float,float]:
    ea = exp_score(ra, rb, home_is_a=True)
    eb = 1.0 - ea
    ra_new = ra + ELO_K * (s_home - ea)
    rb_new = rb + ELO_K * ((1.0 - s_home) - eb)
    return ra_new, rb_new

def regress(r, mean, alpha): return (1-alpha)*r + alpha*mean

def compute_elo_prematch(mdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lid, g in mdf.sort_values("match_date").groupby("league_id"):
        ratings: Dict[int, EloState] = {}
        cur_season = None
        league_mean = ELO_START
        for _, r in g.iterrows():
            season = int(r["season_year"])
            if cur_season is None:
                cur_season = season
            if season != cur_season:
                if ratings:
                    pool = [t.rating for t in ratings.values()]
                    league_mean = float(np.mean(pool)) if pool else ELO_START
                    for tid in ratings:
                        ratings[tid].rating = regress(ratings[tid].rating, league_mean, ELO_SEASON_REGRESS)
                cur_season = season

            h, a = int(r["home_team_id"]), int(r["away_team_id"])
            if h not in ratings: ratings[h] = EloState(ELO_START)
            if a not in ratings: ratings[a] = EloState(ELO_START)

            rows.append({"fixture_id": r["fixture_id"],
                         "elo_home": ratings[h].rating,
                         "elo_away": ratings[a].rating})

            # update post-result
            hg, ag = r["home_goals"], r["away_goals"]
            if pd.isna(hg) or pd.isna(ag): continue
            s_home = 1.0 if hg>ag else (0.5 if hg==ag else 0.0)
            rh, ra = update_elo(ratings[h].rating, ratings[a].rating, s_home)
            ratings[h].rating, ratings[a].rating = rh, ra
    return pd.DataFrame(rows)

elo_df = compute_elo_prematch(matches)
matches = matches.merge(elo_df, on="fixture_id", how="left")
matches["elo_diff"] = matches["elo_home"] - matches["elo_away"]

# -------------------------
# Collapse back to fixture level + diffs
# -------------------------
home_roll = team_match.loc[team_match["team_side"]=="home", [
    "fixture_id","team_id","rest_days",
    "gf_L5","ga_L5","gf_L10","ga_L10",
    "shots_L5","shots_on_target_L5","corners_L5","yellow_cards_L5","red_cards_L5",
    "shots_L10","shots_on_target_L10","corners_L10","yellow_cards_L10","red_cards_L10",
    "xg_L5","xg_L10"
]].rename(columns={
    "team_id":"home_team_id",
    "rest_days":"rest_days_home",
    "gf_L5":"gf_home_L5","ga_L5":"ga_home_L5",
    "gf_L10":"gf_home_L10","ga_L10":"ga_home_L10",
    "shots_L5":"sh_home_L5","shots_on_target_L5":"sot_home_L5","corners_L5":"corn_home_L5",
    "yellow_cards_L5":"yc_home_L5","red_cards_L5":"rc_home_L5",
    "shots_L10":"sh_home_L10","shots_on_target_L10":"sot_home_L10","corners_L10":"corn_home_L10",
    "yellow_cards_L10":"yc_home_L10","red_cards_L10":"rc_home_L10",
    "xg_L5":"xg_home_L5","xg_L10":"xg_home_L10",
})

away_roll = team_match.loc[team_match["team_side"]=="away", [
    "fixture_id","team_id","rest_days",
    "gf_L5","ga_L5","gf_L10","ga_L10",
    "shots_L5","shots_on_target_L5","corners_L5","yellow_cards_L5","red_cards_L5",
    "shots_L10","shots_on_target_L10","corners_L10","yellow_cards_L10","red_cards_L10",
    "xg_L5","xg_L10"
]].rename(columns={
    "team_id":"away_team_id",
    "rest_days":"rest_days_away",
    "gf_L5":"gf_away_L5","ga_L5":"ga_away_L5",
    "gf_L10":"gf_away_L10","ga_L10":"ga_away_L10",
    "shots_L5":"sh_away_L5","shots_on_target_L5":"sot_away_L5","corners_L5":"corn_away_L5",
    "yellow_cards_L5":"yc_away_L5","red_cards_L5":"rc_away_L5",
    "shots_L10":"sh_away_L10","shots_on_target_L10":"sot_away_L10","corners_L10":"corn_away_L10",
    "yellow_cards_L10":"yc_away_L10","red_cards_L10":"rc_away_L10",
    "xg_L5":"xg_away_L5","xg_L10":"xg_away_L10",
})

feat = matches.merge(home_roll, on=["fixture_id","home_team_id"], how="left") \
              .merge(away_roll, on=["fixture_id","away_team_id"], how="left")

# Context
feat["month"] = feat["match_date"].dt.month
feat["home_adv"] = 1
if "round_number" not in feat: feat["round_number"] = np.nan

# Diffs
feat["gf_diff_L5"]   = feat["gf_home_L5"] - feat["gf_away_L5"]
feat["ga_diff_L5"]   = feat["ga_home_L5"] - feat["ga_away_L5"]
feat["sot_diff_L5"]  = feat["sot_home_L5"] - feat["sot_away_L5"]
feat["corn_diff_L5"] = feat["corn_home_L5"] - feat["corn_away_L5"]
feat["xg_diff_L5"]   = feat["xg_home_L5"] - feat["xg_away_L5"]
feat["rest_diff"]    = feat["rest_days_home"] - feat["rest_days_away"]

# Keep labeled rows
feat = feat[feat["outcome"].notna()].reset_index(drop=True)

# -------------------------
# Splits
# -------------------------
train_df = feat[feat["season_year"].isin(TRAIN_YEARS)].copy()
valid_df = feat[feat["season_year"] == VALID_YEAR].copy()
test_df  = feat[feat["season_year"] == TEST_YEAR].copy()

# -------------------------
# Feature lists
# -------------------------
num_cols = [
    "elo_home","elo_away","elo_diff",
    "gf_home_L5","ga_home_L5","gf_away_L5","ga_away_L5",
    "sot_home_L5","sot_away_L5","corn_home_L5","corn_away_L5",
    "xg_home_L5","xg_away_L5",        # if early seasons miss xG, imputer will handle
    "gf_diff_L5","ga_diff_L5","sot_diff_L5","corn_diff_L5","xg_diff_L5",
    "rest_days_home","rest_days_away","rest_diff","round_number","home_adv",
]
opt_cols = [
    "gf_home_L10","ga_home_L10","gf_away_L10","ga_away_L10",
    "sh_home_L5","sh_away_L5","sh_home_L10","sh_away_L10",
    "xg_home_L10","xg_away_L10"
]
num_cols = [c for c in num_cols if c in feat.columns]
opt_cols = [c for c in opt_cols if c in feat.columns]
cat_cols = [c for c in ["league_id","month"] if c in feat.columns]
feature_cols = num_cols + opt_cols + cat_cols

def XY(df):
    X = df[feature_cols].copy()
    y = df["outcome"].map({"H":0,"D":1,"A":2}).astype(int).values
    return X, y

X_train, y_train = XY(train_df)
X_valid, y_valid = XY(valid_df)
X_test,  y_test  = XY(test_df)

# -------------------------
# Pipeline (impute + one-hot + HGB)
# -------------------------
numeric_features = [c for c in feature_cols if c not in cat_cols]
categorical_features = cat_cols

pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ],
    remainder="drop"
)

clf = HistGradientBoostingClassifier(
    loss="log_loss",
    max_iter=300,
    learning_rate=0.04,
    l2_regularization= 0.0,
    random_state=42
)

pipe = Pipeline([("prep", pre), ("clf", clf)])
pipe.fit(X_train, y_train)

# Calibrate on 2023
cal = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
cal.fit(X_valid, y_valid)

# Evaluate
def evaluate(name, model, X, y):
    p = model.predict_proba(X)
    print(f"{name}: log_loss={log_loss(y,p,labels=[0,1,2]):.4f}, acc={ (p.argmax(1)==y).mean():.3f}")

print("\n=== Validation (2023) ===")
evaluate("Valid", cal, X_valid, y_valid)

print("\n=== Test (2024) ===")
evaluate("Test ", cal, X_test, y_test)

# Peek predictions
if len(test_df):
    print("\nSample test probs [H,D,A]:\n", np.round(cal.predict_proba(X_test[:5]), 3))
