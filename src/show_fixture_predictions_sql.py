# src/show_fixture_predictions_sql.py
# from sqlalchemy import create_engine, text
# import os

# eng = create_engine(os.environ["POSTGRES_URL"], pool_pre_ping=True)

# with eng.begin() as c:
#     # 1) Drop old plain columns if they exist (safe if they don't)
#     c.execute(text("""
#         ALTER TABLE standings
#         DROP COLUMN IF EXISTS goal_diff,
#         DROP COLUMN IF EXISTS points,
#         DROP COLUMN IF EXISTS match_played
#     """))

#     # 2) Add them back as GENERATED columns (Postgres will compute & store values)
#     c.execute(text("""
#         ALTER TABLE standings
#         ADD COLUMN goal_diff INTEGER GENERATED ALWAYS AS (goals_for - goals_against) STORED,
#         ADD COLUMN points INTEGER GENERATED ALWAYS AS (3*wins + draws) STORED,
#         ADD COLUMN match_played INTEGER GENERATED ALWAYS AS (wins + draws + losses) STORED
#     """))
# print("standings: generated columns set.")

from sqlalchemy import create_engine, text
import os

eng = create_engine(os.environ["POSTGRES_URL"], pool_pre_ping=True)

ddl = """
DROP TABLE IF EXISTS fixture_predictions CASCADE;
CREATE TABLE fixture_predictions (
  prediction_id     BIGSERIAL PRIMARY KEY,
  fixture_id        INTEGER NOT NULL REFERENCES matches(fixture_id),
  league_id         INTEGER NOT NULL REFERENCES leagues(league_id),
  season_year       INTEGER NOT NULL,
  match_date        DATE    NOT NULL,
  home_team_id      INTEGER NOT NULL REFERENCES teams(team_id),
  away_team_id      INTEGER NOT NULL REFERENCES teams(team_id),
  p_home            DOUBLE PRECISION NOT NULL CHECK (p_home BETWEEN 0 AND 1),
  p_draw            DOUBLE PRECISION NOT NULL CHECK (p_draw BETWEEN 0 AND 1),
  p_away            DOUBLE PRECISION NOT NULL CHECK (p_away BETWEEN 0 AND 1),
  pred_label        TEXT,                 -- plain column
  model_name        TEXT NOT NULL,
  model_version     TEXT NOT NULL DEFAULT '',
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  actual_home_goals INTEGER,
  actual_away_goals INTEGER,
  actual_label      TEXT,                 -- plain column
  UNIQUE (fixture_id, model_name, model_version),
  CHECK ( (p_home + p_draw + p_away) BETWEEN 0.999 AND 1.001 )
);
"""
with eng.begin() as c:
    c.execute(text(ddl))
print("Created fixture_predictions (plain label columns).")
