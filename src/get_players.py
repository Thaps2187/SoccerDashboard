import requests
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)  # read .env file

engine = create_engine("sqlite:////Users/thapeloclement/SoccerDashboard/soccer.db", future=True)

API_KEY = os.getenv("API_KEY")
API_HOST = "v3.football.api-sports.io"


# This collects team's squad 
def get_squad(team_id):

    url = "https://v3.football.api-sports.io/players/squads"

    headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': API_HOST}

    params = {"team": team_id}

    response = requests.get(url, params= params, headers= headers)

    if response.status_code == 200: # code=200 means success
        data = response.json()
        return data
    else:
        print("Error:", response.status_code, response.text)
        return None

 # This is collects player's profile (basic information)   
def get_player(player_id):

    url = "https://v3.football.api-sports.io/players/profiles"

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

    year = 2025


    #Available teams to be collected
    teams_sql = text("""
        SELECT DISTINCT home_team_iD FROM matches
        WHERE season_year = 2025;""")

    teams = engine.connect().execute(teams_sql).scalars().all()


    #SQL quaries to load the data and the file to be added to the table
    squad_sql = text("""
    INSERT INTO squad_membership (
            player_id, team_id, season_year, jersey_number
        ) VALUES (
            :player_id, :team_id, :season_year, :jersey_number
        );
    """)
    profile_sql = text("""
    INSERT INTO players (
            player_id, first_name, last_name, date_of_birth, nationality, position, height, photo_url                
        ) VALUES (
            :player_id, :first_name, :last_name, :date_of_birth, :nationality, :position, :height, :photo_url       
        )
        ON CONFLICT (player_id) DO UPDATE SET
            first_name      = excluded.first_name,
            last_name       = excluded.last_name,
            date_of_birth   = excluded.date_of_birth;
    """)
    
    profiles = []
    squads = []

    player_ids = []

    #Extract the team squad
    for t in teams:
        resp = get_squad(t)

        if not resp or not resp.get("response"):   # empty or missing response
            continue

        players = resp["response"][0]["players"]

        for player in players:
            if player["id"] == 0:
                continue

            squads.append({
                "player_id": player["id"],
                "team_id": t,
                "season_year": year,
                "jersey_number": player["number"]
                })
            
            player_ids.append(player["id"])
    
    
    #Extract player information
    for p in player_ids:
        resp = get_player(p)

        if not resp or not resp.get("response"):   # empty or missing response
            continue

        player = resp["response"][0]["player"]

        profiles.append({
            "player_id": player["id"],
            "first_name": player["firstname"],
            "last_name": player["lastname"],
            "date_of_birth": player["birth"]["date"],
            "nationality": player["nationality"],
            "position": player["position"],
            "height": player["height"],
            "photo_url": player["photo"]
        })


    #Loading data to the database
    with engine.begin() as conn:
        conn.execute(profile_sql, profiles)
        conn.execute(squad_sql, squads)
        
