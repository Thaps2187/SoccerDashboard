import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text, create_engine
from feature_engineering import FeatureEngineer, FEATURE_COLUMNS

engine = create_engine("sqlite:////Users/thapeloclement/SoccerDashboard/soccer.db")

ROOT = Path(__file__).resolve().parents[1] 
MODEL_DIR = ROOT / "models" / "final_K16_HA60_SR30_xg0_L101_rest1_cc0_lr0.04_it300_l20.0_calisotonic"
MODEL_PATH = MODEL_DIR / "model.joblib"

# prefer calibrated model if present
for fname in ["model_calibrated.joblib", "model.joblib", "model.pkl"]:
    p = MODEL_DIR / fname
    if p.exists():
        MODEL_PATH = p
        break
else:
    raise FileNotFoundError(f"No model file found under {MODEL_DIR}")


print("Loading model:", MODEL_PATH)             
model = joblib.load(MODEL_PATH)

MODEL_NAME = "final_K16_HA60_SR30_xg0_L101_rest1_cc0_lr0.04_it300_l20.0_calisotonic"
MODEL_VERSION = "v1"

SEASON_YEAR = 2025

with engine.begin() as conn:
    upcoming = pd.read_sql_query(text("""
        WITH ns AS (
        SELECT fixture_id, league_id, season_year, match_day, match_date,
                home_team_id, away_team_id
        FROM matches
        WHERE season_year = :yr AND status = 'NS' AND match_day IS NOT NULL
        ),
        ns_num AS (
          SELECT ns.*, CAST(REPLACE(UPPER(TRIM(match_day)),'MD','') AS INTEGER) AS md_num
          FROM ns
        ),
        min_md AS (
          SELECT league_id, MIN(md_num) AS md_num
          FROM ns_num
          GROUP BY league_id
        )
        SELECT n.*
        FROM ns_num n
        JOIN min_md m ON m.league_id = n.league_id AND m.md_num = n.md_num
        ORDER BY league_id, match_date;
        """), conn, params={"yr": SEASON_YEAR})

fe = FeatureEngineer(engine, form_n=5, h2h_n=5)
X = fe.build(upcoming)

cols = [
    "gf_home_L5","ga_home_L5","gf_away_L5","ga_away_L5",
    "gf_home_L10","ga_home_L10","gf_away_L10","ga_away_L10",
    "sh_home_L5","sh_away_L5","sot_home_L5","sot_away_L5",
    "rest_days_home","rest_days_away","rest_diff",
    "elo_home","elo_away","elo_diff",
    "h2h_home_winrate",
]

# print(sorted(X.columns.tolist()))
# missing = [c for c in cols if c not in X.columns]
# present = [c for c in cols if c in X.columns]
# print("Missing:", missing)
# print("Present:", present)
# if present:
#     print(X[present].describe())

proba = model.predict_proba(X)

proba = model.predict_proba(X)

iH,iD,iA = 0,1,2
p_home, p_draw, p_away = proba[:, iH], proba[:, iD], proba[:, iA]

UPSERT_SQL = text("""
INSERT INTO fixture_predictions
    (fixture_id, league_id, season_year, match_date,
    home_team_id, away_team_id,
    p_home, p_draw, p_away,
    model_name, model_version)
VALUES
    (:fixture_id, :league_id, :season_year, :match_date,
    :home_team_id, :away_team_id,
    :p_home, :p_draw, :p_away,
    :model_name, :model_version)
ON CONFLICT(fixture_id, model_name, model_version) DO UPDATE SET
    p_home     = excluded.p_home,
    p_draw     = excluded.p_draw,
    p_away     = excluded.p_away,
    match_date = excluded.match_date
""")

records = []
for i, r in upcoming.iterrows():
    records.append({
        "fixture_id":   int(r.fixture_id),
        "league_id":    int(r.league_id),
        "season_year":  int(r.season_year),
        "match_date":   str(r.match_date),   
        "home_team_id": int(r.home_team_id),
        "away_team_id": int(r.away_team_id),
        "p_home": float(p_home[i]),
        "p_draw": float(p_draw[i]),
        "p_away": float(p_away[i]),
        "model_name":   MODEL_NAME,
        "model_version":MODEL_VERSION
    })
print(f"Total Number of predictions {len(records)}")
with engine.begin() as conn:
    conn.execute(UPSERT_SQL, records)
