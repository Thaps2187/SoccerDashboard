import requests
import pandas as pd
import os, time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)  # read .env file

stats_sql = text("""
INSERT INTO match_stats (
    fixture_id, home_xgoals, away_xgoals, home_corners, away_corners,
    home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards, home_shots, away_shots,
    home_shots_on_goal, away_shots_on_goal, home_gk_saves, away_gk_saves,
    home_possession, away_possession, home_pass_accuracy, away_pass_accuracy,
    home_offsides, away_offsides, home_fouls, away_fouls
) VALUES (
    :fixture_id, :home_xgoals, :away_xgoals, :home_corners, :away_corners,
    :home_yellow_cards, :away_yellow_cards, :home_red_cards, :away_red_cards, :home_shots, :away_shots,
    :home_shots_on_goal, :away_shots_on_goal, :home_gk_saves, :away_gk_saves,
    :home_possession, :away_possession, :home_pass_accuracy, :away_pass_accuracy,
    :home_offsides, :away_offsides, :home_fouls, :away_fouls
)
ON CONFLICT(fixture_id) DO UPDATE SET
    home_xgoals        = excluded.home_xgoals,
    away_xgoals        = excluded.away_xgoals,
    home_corners       = excluded.home_corners,
    away_corners       = excluded.away_corners,
    home_yellow_cards  = excluded.home_yellow_cards,
    away_yellow_cards  = excluded.away_yellow_cards,
    home_red_cards     = excluded.home_red_cards,
    away_red_cards     = excluded.away_red_cards,
    home_shots         = excluded.home_shots,
    away_shots         = excluded.away_shots,
    home_shots_on_goal = excluded.home_shots_on_goal,
    away_shots_on_goal = excluded.away_shots_on_goal,
    home_gk_saves      = excluded.home_gk_saves,
    away_gk_saves      = excluded.away_gk_saves,
    home_possession    = excluded.home_possession,
    away_possession    = excluded.away_possession,
    home_pass_accuracy = excluded.home_pass_accuracy,
    away_pass_accuracy = excluded.away_pass_accuracy,
    home_offsides      = excluded.home_offsides,
    away_offsides      = excluded.away_offsides,
    home_fouls         = excluded.home_fouls,
    away_fouls         = excluded.away_fouls;
""")

empty_stats = text("""INSERT INTO match_stats (fixture_id, has_stats) VALUES (:id, 0)
                      ON CONFLICT(fixture_id) DO UPDATE SET has_stats=0;"""
                 )

engine = create_engine("sqlite:////Users/thapeloclement/SoccerDashboard/soccer.db", future=True)

API_KEY = os.getenv("API_KEY")
API_HOST = "v3.football.api-sports.io"

def get_statistics(fixture_id):

    url = "https://v3.football.api-sports.io/fixtures/statistics"

    headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': API_HOST}

    params = {"fixture": fixture_id}

    response = requests.get(url, params= params, headers= headers)

    if response.status_code == 200: # code=200 means success
        data = response.json()
        return data
    else:
        print("Error:", response.status_code, response.text)
        return None
    
def parse_statistics(data):
    def find_stat(stats_list, name):
        for s in stats_list:
            if s["type"] == name:
                return s["value"] or 0
        return 0
    
    if not data or "response" not in data or len(data["response"]) < 2:
        return None
    
    home = data["response"][0]
    away = data["response"][1]

    return {
        "fixture_id": data["parameters"]["fixture"],
        "home_xgoals": find_stat(home["statistics"], "expected_goals"),
        "away_xgoals": find_stat(away["statistics"], "expected_goals"),
        "home_corners": find_stat(home["statistics"], "Corner Kicks"),
        "away_corners": find_stat(away["statistics"], "Corner Kicks"),
        "home_yellow_cards": find_stat(home["statistics"], "Yellow Cards"),
        "away_yellow_cards": find_stat(away["statistics"], "Yellow Cards"),
        "home_red_cards": find_stat(home["statistics"], "Red Cards"),
        "away_red_cards": find_stat(away["statistics"], "Red Cards"),
        "home_shots": find_stat(home["statistics"], "Total Shots"),
        "away_shots": find_stat(away["statistics"], "Total Shots"),
        "home_shots_on_goal": find_stat(home["statistics"], "Shots on Goal"),
        "away_shots_on_goal": find_stat(away["statistics"], "Shots on Goal"),
        "home_gk_saves": find_stat(home["statistics"], "Goalkeeper Saves"),
        "away_gk_saves": find_stat(away["statistics"], "Goalkeeper Saves"),
        "home_possession": find_stat(home["statistics"], "Ball Possession"),
        "away_possession": find_stat(away["statistics"], "Ball Possession"),
        "home_pass_accuracy": find_stat(home["statistics"], "Passes %"),
        "away_pass_accuracy": find_stat(away["statistics"], "Passes %"),
        "home_offsides": find_stat(home["statistics"], "Offsides"),
        "away_offsides": find_stat(away["statistics"], "Offsides"),
        "home_fouls": find_stat(home["statistics"], "Fouls"),
        "away_fouls": find_stat(away["statistics"], "Fouls")
    }

if __name__ == "__main__":


    #Getting fixtures for the from 2015 till 2024 to be used in predictive modelling
    get_fixture_txt = text(
        """ 
        SELECT fixture_id FROM matches
        WHERE season_year = 2025 AND status = 'FT';
        """)
    
    fixtures = engine.connect().execute(get_fixture_txt).scalars().all()

    # # Load Existing stats
    # stats_file = "data/bundasliga_stats.csv"
    # if os.path.exists(stats_file):
    #     stats_df = pd.read_csv(stats_file)
    #     done_ids =set(stats_df["fixture_id"].values)
    # else:
    #     stats_df = pd.DataFrame()
    #     done_ids = set()
    
    match_stats = []
    #missing_fx = []

    for f in fixtures:
        
        data = get_statistics(f)

        if (not data) or ("response" not in data) or (len(data["response"]) < 2):
            missing_fx.append(f)
            continue
    
        parsed = parse_statistics(data)

        if parsed:
            match_stats.append(parsed)
        

    #missing_fx = [{"id": i} for i in missing_fx]

    with engine.begin() as conn:
            conn.execute(stats_sql, match_stats) 
            #conn.execute(empty_stats, missing_fx)

    # if match_stats:
    #     new_df = pd.DataFrame(match_stats)
    #     final_df = pd.concat([stats_df, new_df], ignore_index=True)
    #     final_df.to_csv(stats_file, index=False)
    #     print(f"✅ Saved {len(match_stats)} new stats (total {len(final_df)})")

