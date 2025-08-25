import requests
import pandas as pd
import os, time

API_KEY = "86e06e06acmsh6f3ca20dc563908p19bdc2jsn2bdcdce00e02"
API_HOST = "api-football-v1.p.rapidapi.com"

def get_statistics(fixture_id):

    url = f"https://{API_HOST}/v3/fixtures/statistics"
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
        "home_XGoals": find_stat(home["statistics"], "expected_goals"),
        "away_XGoals": find_stat(away["statistics"], "expected_goals"),
        "home_corners": find_stat(home["statistics"], "Corner Kicks"),
        "away_corners": find_stat(away["statistics"], "Corner Kicks"),
        "home_yellow_cards": find_stat(home["statistics"], "Yellow Cards"),
        "away_yellow_cards": find_stat(away["statistics"], "Yellow Cards"),
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
    fixtures = pd.read_csv("data/epl_fixtures_2024.csv")

    # Load Existing stats
    stats_file = "data/epl_stats.csv"
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        done_ids =set(stats_df["fixture_id"].values)
    else:
        stats_df = pd.DataFrame()
        done_ids = set()
    
    # Filter fixtures without stats
    pending = fixtures[~fixtures["fixture_id"].isin(done_ids)]
    print(f"{len(pending)} fixtures without stats")

    new_rows = []

    for i, row in pending.head(90).iterrows():          #limiting to 80 calls a day
        print(f"Fetching stats for {row["fixture_id"]}...")
        data = get_statistics(row["fixture_id"])
        parsed = parse_statistics(data)

        if parsed:
            new_rows.append(parsed)
        
        time.sleep(15) # To avoid hitting API limits

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([stats_df, new_df], ignore_index=True)
        final_df.to_csv(stats_file, index=False)
        print(f"✅ Saved {len(new_rows)} new stats (total {len(final_df)})")

