import requests
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)  # read .env file

engine = create_engine("sqlite:////Users/thapeloclement/SoccerDashboard/soccer.db", future=True)

API_KEY = os.getenv("API_KEY")
API_HOST = "v3.football.api-sports.io"

def get_player_season(player_id):

    url = "https://v3.football.api-sports.io/players/seasons"

    headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': API_HOST}

    params = {"player": player_id}

    response = requests.get(url, params= params, headers= headers)

    if response.status_code == 200: # code=200 means success
        data = response.json()
        return data
    else:
        print("Error:", response.status_code, response.text)
        return None
    


if __name__ == "__main__":

    player_sql = text("""
        SELECT player_id FROM players;
    """)
    player_season_sql = text("""
        INSERT INTO player_seasons (player_id, season_year)
        VALUES (:player_id, :season_year)
        ON CONFLICT(player_id, season_year) DO NOTHING;
    """)

    player_ids = engine.connect().execute(player_sql).scalars().all()

    player_season = []

    for player in player_ids:
        resp = get_player_season(player)

        if not resp or not resp.get("response"):   # empty or missing response
            continue

        seasons = resp["response"]

        for s in seasons:
            player_season.append({
                "player_id": player,
                "season_year": s
            })
        
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = ON;")
            conn.execute(player_season_sql, player_season) 


