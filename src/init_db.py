from sqlalchemy import create_engine, text

DB_PATH = "sqlite:///soccer.db"

engine = create_engine(DB_PATH)

with engine.begin() as conn:
    conn.exec_driver_sql("PRAGMA foreign_keys = ON;")

    #Leagues
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS leagues (
        league_id   INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        logo_url    TEXT,
        country     TEXT
    );
    """))
    
    #Teams
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        logo_url TEXT,
        country TEXT
    );
    """))

    # Matches
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS matches (
        fixture_id      INTEGER PRIMARY KEY,
        league_id       INTEGER NOT NULL,
        season_year     INTEGER NOT NULL,
        season          TEXT,
        matchday_text   TEXT,
        match_date      TEXT,
        home_team_id    INTEGER NOT NULL,
        away_team_id    INTEGER NOT NULL,
        referee         TEXT,
        home_goals      INTEGER,
        away_goals      INTEGER,
        home_results    TEXT,
        away_results    TEXT,
    
        FOREIGN KEY (league_id)     REFERENCES leagues(league_id),
        FOREIGN KEY (home_team_id)  REFERENCES teams(team_id),
        FOREIGN KEY (away_team_id)  REFERENCES teams(team_id)
    );
    """))

    #Match Statistics
    conn.execute(text(""" 
    CREATE TABLE IF NOT EXISTS match_stats (
        fixture_id INTEGER PRIMARY KEY,       -- same as in matches
        home_xgoals REAL,
        away_xgoals REAL,
        home_corners INTEGER,
        away_corners INTEGER,
        home_yellow_cards INTEGER,
        away_yellow_cards INTEGER,
        home_red_cards INTEGER,
        away_red_cards INTEGER,
        home_shots INTEGER,
        away_shots INTEGER,
        home_shots_on_goal INTEGER,
        away_shots_on_goal INTEGER,
        home_gk_saves INTEGER,
        away_gk_saves INTEGER,
        home_possession REAL,
        away_possession REAL,
        home_pass_accuracy REAL,
        away_pass_accuracy REAL,
        home_offsides INTEGER,
        away_offsides INTEGER,
        home_fouls INTEGER,
        away_fouls INTEGER,
                        
        FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);
"""
    ))

    #Player basic information
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        date_of_birth TEXT,
        nationality TEXT,
        position TEXT,
        height INTEGER,
        photo_url TEXT
        );
"""))
    
    #Team squads
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS squad_membership (
        membership_id INTEGER PRIMARY KEY,
        player_id INTEGER NOT NULL,
        team_id INTEGER NOT NULL,
        season_year INTEGER,
        jersey_number INTEGER,
                      
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
        );
"""))
    
    #Teams standings after each game for all the matches collected
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS standings (
        league_id INTEGER NOT NULL,
        season_year INTEGER NOT NULL,
        season TEXT NOT NULL,
        round_label TEXT NOT NULL,
        team_id INTEGER NOT NULL,
    
        rank INTEGER NOT NULL,
        wins INTEGER NOT NULL DEFAULT 0,
        draws INTEGER NOT NULL DEFAULT 0,
        losses INTEGER NOT NULL DEFAULT 0,
        goals_for INTEGER NOT NULL DEFAULT 0,
        goals_against INTEGER NOT NULL DEFAULT 0,
        goal_diff INTEGER GENERATED ALWAYS AS (goals_for - goals_against) STORED,
        points INTEGER GENERATED ALWAYS AS (3*wins + 1*draws) STORED,
        match_played  INTEGER GENERATED ALWAYS AS (wins + draws + losses) STORED,
        form TEXT,
                      
        PRIMARY KEY (league_id, season_year, round_label, team_id),
                      
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (league_id) REFERENCES leagues(league_id)
        );
"""))
    
#     conn.execute(text("""
#     CREATE TABLE IF NOT EXISTS player_seasons (
#         player_id INTEGER NOT NULL,
#         season_year INTEGER NOT NULL,
                      
#         PRIMARY KEY (player_id, season_year)
                    
#         FOREIGN KEY (player_id) REFERENCES players(player_id)
#         );
# """))
    # Predicted outcomes for upcoming and past fixtures
    conn.execute(text("""
CREATE TABLE IF NOT EXISTS fixture_predictions (
    prediction_id    INTEGER PRIMARY KEY,
    fixture_id       INTEGER NOT NULL,           
    league_id        INTEGER NOT NULL,
    season_year      INTEGER NOT NULL,
    match_date       TEXT    NOT NULL,           
    home_team_id     INTEGER NOT NULL,
    away_team_id     INTEGER NOT NULL,

    -- probabilities (0..1)
    p_home           REAL    NOT NULL,
    p_draw           REAL    NOT NULL,
    p_away           REAL    NOT NULL,
                      
    pred_label TEXT GENERATED ALWAYS AS (
    CASE
      WHEN p_home >= p_draw AND p_home >= p_away THEN 'H'
      WHEN p_draw >= p_home AND p_draw >= p_away THEN 'D'
      ELSE 'A'
    END
    ) STORED,

    -- model/run metadata
    model_name       TEXT    NOT NULL,           
    model_version    TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),

    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_label TEXT GENERATED ALWAYS AS (
    CASE
      WHEN actual_home_goals IS NULL OR actual_away_goals IS NULL THEN NULL
      WHEN actual_home_goals >  actual_away_goals THEN 'H'
      WHEN actual_home_goals =  actual_away_goals THEN 'D'
      ELSE 'A'
    END
    ) STORED,

    -- data quality guards
    CHECK (p_home BETWEEN 0 AND 1),
    CHECK (p_draw BETWEEN 0 AND 1),
    CHECK (p_away BETWEEN 0 AND 1),
    CHECK (ABS((p_home + p_draw + p_away) - 1.0) <= 0.001),
                      
    UNIQUE (fixture_id, model_name, model_version),

    FOREIGN KEY (home_team_id)  REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id)  REFERENCES teams(team_id),
    FOREIGN KEY (fixture_id)    REFERENCES matches(fixture_id)
    FOREIGN KEY (league_id)     REFERENCES leagues(league_id)
);
"""))
    
