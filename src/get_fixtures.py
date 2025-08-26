import requests
import pandas as pd
import re, os, time
from dotenv import load_dotenv


#################################################################################
#                                                                               #
# This library makes an API call to get fixtures for a provided League ID and   #
#  and season year, the data is then stored in an CSV file                      #
#                                                                               #
#################################################################################

load_dotenv()  # read .env file

API_KEY = os.getenv("API_KEY")
API_HOST = "api-football-v1.p.rapidapi.com"

def get_fixtures(league_id, season):

    url = f"https://{API_HOST}/v3/fixtures"
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
    
    league_id = 39

    # This pulls fixtures from the last few years starting from year 2010 till 2024
    seasons = [2010, 2011, 2012, 2013, 2014, 2015, 2016,
              2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    def results(team_1, team_2):
        if team_1 > team_2:
            return "W"          #Win
        elif team_1 < team_2:
            return "L"          #Lose
        else:
            return "D"          #Draw
    
    os.makedirs("data_epl", exist_ok=True)


    for season in seasons:

        print(f"Pulling season {season}…")

        # First page
        data = get_fixtures(league_id, season)
        if not data:
            print(f"No data returned for {season}")
            continue

        # Handle paging 
        responses = data.get("response", [])
        paging = data.get("paging", {}) or {}
        current = paging.get("current", 1)
        total = paging.get("total", 1)

        # If the API paginates, fetch the rest
        while current < total:
            current += 1
            
            url = f"https://{API_HOST}/v3/fixtures"
            headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': API_HOST}
            params = {"league": league_id, "season": season, "page": current}
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                more = r.json()
                responses.extend(more.get("response", []))
                time.sleep(0.15)  # tiny pause
            else:
                print(f"⚠️  Page {current} error {r.status_code}: {r.text}")
                break


        rows = []
        for f in responses:
            #Safe match_day extraction (handles missing/odd rounds)
            round_str = (f.get("league", {}) or {}).get("round") or ""
            md = re.findall(r"(\d+)", round_str)
            match_day = int(md[0]) if md else None

            rows.append({
                "fixture_id": f["fixture"]["id"],
                "season": f"{season}/{((season % 100) + 1)}",
                "date": f["fixture"]["date"],
                "match_day": match_day,
                "referee": f["fixture"]["referee"],
                "home_team": f["teams"]["home"]["name"],
                "away_team": f["teams"]["away"]["name"],
                "home_goals": f["goals"]["home"],
                "away_goals": f["goals"]["away"],
                "home_results": results(f["goals"]["home"], f["goals"]["away"]),
                "away_results": results(f["goals"]["away"], f["goals"]["home"]),
            })
            
        df = pd.DataFrame(rows)

        # If nothing parsed, skip this season cleanly
        if df.empty:
            print(f"No parsed rows for season {season} (skipping save).")
            continue

        # Make sure 'date' exists and is sortable
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", na_position="last")
        else:
            print(f" 'date' not found for season {season}; saving unsorted.")

        df.sort_values("date", inplace=True)
        out_path = f"data/epl_fixtures_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(df)} fixtures for season {season} -> {out_path}")

    print("Done Done.")
