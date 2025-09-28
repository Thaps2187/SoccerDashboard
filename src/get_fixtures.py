import requests
import pandas as pd
import re, os, time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


#################################################################################
#                                                                               #
# This library makes an API call to get fixtures for a provided League ID and   #
#  and season year, the data is then stored in an CSV file                      #
#                                                                               #
#################################################################################

load_dotenv(override=True)  # read .env file

engine = create_engine("sqlite:////Users/thapeloclement/SoccerDashboard/soccer.db", future=True)

API_KEY = os.getenv("API_KEY")
API_HOST = "v3.football.api-sports.io"

def get_fixtures(league_id, season):

    url = "https://v3.football.api-sports.io/fixtures"
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
    
    league_ids = [39, 61, 78, 135, 140]

    # This pulls fixtures from the last few years starting from year 2010 till 2025
    seasons = [2025]

    def results(team_1, team_2):
        # if either score is missing, we can't compute a result yet
        if team_1 is None or team_2 is None:
            return None
        if team_1 > team_2:
            return "W"          #Win
        elif team_1 < team_2:
            return "L"          #Lose
        else:
            return "D"          #Draw
    
    os.makedirs("data_epl", exist_ok=True)

    for league_id in league_ids:

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
                
                url = "https://v3.football.api-sports.io/fixtures"
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

            # ---------- 1) Build unique teams payload ----------
            teams = {}
            for f in responses:
                h = f["teams"]["home"]
                a = f["teams"]["away"]
                for t in (h, a):
                    teams[t["id"]] = {
                        "team_id":   t["id"],
                        "name":      t["name"],
                        "country":   f["league"].get("country"),              
                        "logo_url":  t.get("logo"),
                    }
            teams_payload = list(teams.values())

            # ---------- 2) Upsert teams ----------
            sql_upsert_teams = text("""
            INSERT INTO teams (team_id, name, country, logo_url)
            VALUES (:team_id, :name, :country, :logo_url)
            ON CONFLICT(team_id) DO UPDATE SET
            name     = excluded.name,
            country  = COALESCE(excluded.country, teams.country),
            logo_url = COALESCE(excluded.logo_url, teams.logo_url);
            """)

            # ---------- 3) Build matches payload ----------
            matches_payload = []
            for f in responses:
                #Safe match_day extraction (handles missing/odd rounds)
                round_str = (f.get("league", {}) or {}).get("round") or ""
                md = re.findall(r"(\d+)", round_str)
                match_day = f"MD{int(md[0])}" if md else None


                fx   = f["fixture"]
                lg   = f["league"]
                home = f["teams"]["home"]
                away = f["teams"]["away"]
                goals = f.get("goals", {})

                matches_payload.append({
                    "fixture_id":   fx["id"],
                    "league_id":    lg["id"],
                    "season_year":  lg["season"],
                    "season":       f"{season}/{((season % 100) + 1)}",
                    "match_date":   fx.get("date"),
                    "match_day":    match_day,
                    "status":       fx["status"]["short"],
                    "referee":      fx["referee"].split(",")[0].strip() if fx.get("referee") else None,
                    "home_team_id": home["id"],
                    "away_team_id": away["id"],
                    "home_goals":   goals.get("home"),
                    "away_goals":   goals.get("away"),
                    "home_results": results(goals.get("home"), goals.get("away")),
                    "away_results": results(goals.get("away"), goals.get("home")),
                })

            sql_upsert_matches = text("""
            INSERT INTO matches (
            fixture_id, league_id, season_year, season, match_day, match_date, status, referee,
            home_team_id, away_team_id, home_goals, away_goals, home_results, away_results
            ) VALUES (
            :fixture_id, :league_id, :season_year, :season, :match_day, :match_date, :status, :referee,
            :home_team_id, :away_team_id, :home_goals, :away_goals, :home_results,
            :away_results
            )
            ON CONFLICT(fixture_id) DO UPDATE SET
            status       = excluded.status,
            home_goals   = excluded.home_goals,
            away_goals   = excluded.away_goals,
            home_results = excluded.home_results,
            away_results = excluded.away_results,
            referee      = COALESCE(excluded.referee, matches.referee),
            -- Only updating where Status are now FT
            match_day    = matches.match_day,
            match_date   = matches.match_date
            WHERE matches.status = 'NS'
            AND excluded.status IN ('FT','AET','PEN');
            """)

            # ---------- 4) Write to DB in the right order ----------
            with engine.begin() as conn:
                #conn.execute(sql_upsert_teams, teams_payload)     # parents first
                conn.execute(sql_upsert_matches, matches_payload) # then children

            print(f"Upserted {len(matches_payload)} teams and {len(matches_payload)} fixtures.")

                
            # df = pd.DataFrame(matches_payload)

            # # If nothing parsed, skip this season cleanly
            # if df.empty:
            #     print(f"No parsed rows for season {season} (skipping save).")
            #     continue

            # # Make sure 'date' exists and is sortable
            # if "date" in df.columns:
            #     df["date"] = pd.to_datetime(df["date"], errors="coerce")
            #     df = df.sort_values("date", na_position="last")
            # else:
            #     print(f" 'date' not found for season {season}; saving unsorted.")

            # df.sort_values("date", inplace=True)
            # out_path = f"data/l1_fixtures/l1_fixtures_{season}.csv"
            # df.to_csv(out_path, index=False)
            # print(f"✅ Saved {len(df)} fixtures for season {season} -> {out_path}")

    print("Done Done.")
