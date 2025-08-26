import pandas as pd
import requests, os
from dotenv import load_dotenv

load_dotenv()  # read .env file

API_KEY = os.getenv("API_KEY")
API_HOST = "api-football-v1.p.rapidapi.com"

def get_standings(league_id, season):

    url = f"https://{API_HOST}/v3/standings"
    headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': API_HOST}

    params = {"league": league_id, "season": season}

    response = requests.get(url, params= params, headers= headers)

    if response.status_code == 200: # code=200 means success
        data = response.json()
        return data
    else:
        print("Error:", response.status_code, response.text)
        return None

if __name__ == "__main__":
    league_id = 61

    # This pulls league from the last few years starting from year 2010 till 2024
    seasons = [2010, 2011, 2012, 2013, 2014, 2015, 2016,
              2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    for season in seasons:

        print(f"Pulling season {season}…")

        # First page
        data = get_standings(league_id, season)
        if not data:
            print(f"No data returned for {season}")
            continue

        data = data.get("response", [])
        no_of_teams = len(data[0]["league"]["standings"][0])
        print(f"number of teams {no_of_teams}")
        current = 0

        standings = data[0]["league"]["standings"][0]

        rows = []

        while current < no_of_teams:


            rows.append({
                        "Rank": standings[current]["rank"],
                        "Logo": standings[current]["team"]["logo"],
                        "Team": standings[current]["team"]["name"],
                        "MP": standings[current]["all"]["played"],
                        "W": standings[current]["all"]["win"],
                        "D": standings[current]["all"]["draw"],
                        "L": standings[current]["all"]["lose"],
                        "GF": standings[current]["all"]["goals"]["for"],
                        "GA": standings[current]["all"]["goals"]["against"],
                        "GD": standings[current]["goalsDiff"],
                        "Pts": standings[current]["points"],
                        "Form": standings[current]["form"],
                        "Qualification": standings[current]["description"],
                        "Home_wins": standings[current]["home"]["win"],
                        "Home_draw": standings[current]["home"]["draw"],
                        "Home_lose": standings[current]["home"]["lose"],
                        "Home_goals_for": standings[current]["home"]["goals"]["for"],
                        "Home_goals_against": standings[current]["home"]["goals"]["against"],
                        "Away_wins": standings[current]["away"]["win"],
                        "Away_draw": standings[current]["away"]["draw"],
                        "Away_lose": standings[current]["away"]["lose"],
                        "Away_goals_for": standings[current]["away"]["goals"]["for"],
                        "Away_goals_against": standings[current]["away"]["goals"]["against"],
                    })


            current += 1

        df = pd.DataFrame(rows)

        # If nothing parsed, skip this season cleanly
        if df.empty:
            print(f"No parsed rows for season {season} (skipping save).")
            continue

        out_path = f"data/l1_tables/l1_table_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(df)} league standings for season {season} -> {out_path}")
        
    print("Done Done")

