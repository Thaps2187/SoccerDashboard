import requests

API_KEY = "86e06e06acmsh6f3ca20dc563908p19bdc2jsn2bdcdce00e02"
API_HOST = "api-football-v1.p.rapidapi.com"
LEAGUE_ID = 39  # Premier League

def get_available_seasons(league_id):
    url = f"https://{API_HOST}/v3/leagues"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST
    }
    params = {"id": league_id}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    league_info = data.get("response", [{}])[0]
    seasons = [s["year"] for s in league_info.get("seasons", [])]
    return sorted(seasons)

if __name__ == "__main__":
    seasons_list = get_available_seasons(LEAGUE_ID)
    print("EPL seasons available in API:", seasons_list)

# Ran this and found that these are the available seasons:
# EPL seasons available in API: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
#  2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
