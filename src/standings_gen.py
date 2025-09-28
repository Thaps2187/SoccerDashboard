from sqlalchemy import create_engine, text

SQL = """
/* Build round-by-round standings from matches and write into standings */
WITH m AS (
  SELECT
    season_year,
    league_id,
    COALESCE(season, CAST(season_year AS TEXT)) AS season,
    CAST(SUBSTR(UPPER(TRIM(match_day)), 3) AS INTEGER) AS rno,   -- 'MD12' -> 12
    home_team_id, away_team_id,
    COALESCE(home_goals, 0) AS home_goals,
    COALESCE(away_goals, 0) AS away_goals,
    UPPER(TRIM(home_results)) AS hr,      -- W / D / L
    UPPER(TRIM(away_results)) AS ar,      -- W / D / L
    UPPER(TRIM(status))       AS status   -- expect 'FT'
  FROM matches
),
team_rows AS (  -- one row per team per finished match
  SELECT
    season_year, league_id, season, rno,
    home_team_id AS team_id,
    home_goals   AS gf, away_goals AS ga,
    CASE WHEN hr = 'W' THEN 1 ELSE 0 END AS w,
    CASE WHEN hr = 'D' THEN 1 ELSE 0 END AS d,
    CASE WHEN hr = 'L' THEN 1 ELSE 0 END AS l,
    CASE WHEN hr IN ('W','D','L') THEN hr ELSE NULL END AS res
  FROM m WHERE status = 'FT' AND rno IS NOT NULL

  UNION ALL

  SELECT
    season_year, league_id, season, rno,
    away_team_id,
    away_goals, home_goals,
    CASE WHEN ar = 'W' THEN 1 ELSE 0 END,
    CASE WHEN ar = 'D' THEN 1 ELSE 0 END,
    CASE WHEN ar = 'L' THEN 1 ELSE 0 END,
    CASE WHEN ar IN ('W','D','L') THEN ar ELSE NULL END
  FROM m WHERE status = 'FT' AND rno IS NOT NULL
),
per_round AS (  -- cumulative stats + last-5 form
  SELECT
    season_year, league_id, season, team_id, rno,
    SUM(w)  OVER (PARTITION BY season_year, league_id, team_id
                  ORDER BY rno ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS wins,
    SUM(d)  OVER (PARTITION BY season_year, league_id, team_id
                  ORDER BY rno ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS draws,
    SUM(l)  OVER (PARTITION BY season_year, league_id, team_id
                  ORDER BY rno ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS losses,
    SUM(gf) OVER (PARTITION BY season_year, league_id, team_id
                  ORDER BY rno ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS goals_for,
    SUM(ga) OVER (PARTITION BY season_year, league_id, team_id
                  ORDER BY rno ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS goals_against,
    -- If this windowed group_concat errors on your version, remove it and the `form` column below
    group_concat(res, '') OVER (
      PARTITION BY season_year, league_id, team_id
      ORDER BY rno
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS form5
  FROM team_rows
),
ranked AS (  -- rank within (season, league, round)
  SELECT
    season_year, league_id, season, rno, team_id,
    wins, draws, losses, goals_for, goals_against, form5,
    ROW_NUMBER() OVER (
      PARTITION BY season_year, league_id, rno
      ORDER BY
        (3*wins + draws) DESC,              -- points
        (goals_for - goals_against) DESC,   -- goal diff
        goals_for DESC,
        team_id ASC
    ) AS rnk
  FROM per_round
)
INSERT OR REPLACE INTO standings (
  league_id, season_year, season, round_label, team_id,
  rank, wins, draws, losses, goals_for, goals_against, form
)
SELECT
  league_id,
  season_year,
  season,
  printf('MD%d', rno) AS round_label,
  team_id,
  rnk, wins, draws, losses, goals_for, goals_against,
  form5
FROM ranked;
"""

def refresh_standings(db_path="/Users/thapeloclement/SoccerDashboard/soccer.db"):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text(SQL))
    print("Standings refreshed ")

if __name__ == "__main__":
    refresh_standings() 




