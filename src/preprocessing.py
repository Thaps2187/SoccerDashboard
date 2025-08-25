import pandas as pd
from sklearn.preprocessing import LabelEncoder

def compute_team_features(df, team_col, ground):
    results = []

    for team in df[team_col].unique():
        team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
        team_matches = team_matches.sort_values("date")

        # Statistics of interest
        team_matches[f"{ground}_goals"] = team_matches.apply(
            lambda x: x["home_goals"] if x["home_team"] == team else x["away_goals"], axis = 1
            )
        team_matches[f"{ground}_conceded"] = team_matches.apply(
            lambda x: x["away_goals"] if x["away_team"] == team else x["home_goals"], axis = 1
            )
        team_matches[f"{ground}_shots"] = team_matches.apply(
            lambda x: x["home_shots"] if x["home_team"] == team else x["away_shots"], axis = 1
            )
        team_matches[f"{ground}_corners"] = team_matches.apply(
            lambda x: x["home_corners"] if x["home_team"] == team else x["away_corners"], axis = 1
            )
        team_matches[f"{ground}_yellow_cards"] = team_matches.apply(
            lambda x: x["home_yellow_cards"] if x["home_team"] == team else x["away_yellow_cards"], axis = 1
            )
        team_matches[f"{ground}_possession"] = team_matches.apply(
            lambda x: x["home_possession"] if x["home_team"] == team else x["away_possession"], axis=1
            )
        team_matches[f"{ground}_shots_on_goal"] = team_matches.apply(
            lambda x: x["home_shots_on_goal"] if x["home_team"] == team else x["away_shots_on_goal"], axis=1
            )
        team_matches[f"{ground}_fouls_committed"] = team_matches.apply(
            lambda x: x["home_fouls"] if x["home_team"] == team else x["away_fouls"], axis=1
            )
        team_matches[f"{ground}_fouls_conceded"] = team_matches.apply(
            lambda x: x["away_fouls"] if x["away_team"] == team else x["home_fouls"], axis=1
            )
        team_matches[f"{ground}_pass_accuracy"] = team_matches.apply(
            lambda x: x["home_pass_accuracy"] if x["home_team"] == team else x["away_pass_accuracy"], axis=1
            )
        
        # Points per match 
        team_matches[f"{ground}_points"] = team_matches.apply(
            lambda x: 3 if x[f"{ground}_goals"] > x[f"{ground}_conceded"] else 1 if x[f"{ground}_goals"] == x[f"{ground}_conceded"] else 0, axis=1
            )
        
        ## Average for 5 previous matches
        # before starting to convert percentage columns to numeric
        for stat in team_matches.columns:
            if "pass_accuracy" in stat or "possession" in stat:
                team_matches[stat] = team_matches[stat].str.replace('%', '').astype(float)


        stats_to_avg = ["goals", "conceded", "shots", "corners", "yellow_cards", "points", 
                        "possession", "shots_on_goal", "fouls_committed", "fouls_conceded", "pass_accuracy"]
        for stat in stats_to_avg:
            team_matches[f"{ground}_avg_{stat}_5"] = (
                team_matches[f"{ground}_{stat}"].shift().rolling(5, min_periods=1).mean()
            )
        
        team_matches[f"{ground}_wins"] = (team_matches[f"{ground}_goals"] > team_matches[f"{ground}_conceded"]).astype(int)
        team_matches[f"{ground}_draws"] = (team_matches[f"{ground}_goals"] == team_matches[f"{ground}_conceded"]).astype(int)
        team_matches[f"{ground}_losses"] = (team_matches[f"{ground}_goals"] < team_matches[f"{ground}_conceded"]).astype(int)

        for result in ["wins", "draws", "losses"]:
            team_matches[f"{ground}_avg_{result}_5"] = (
                team_matches[f"{ground}_{result}"].shift().rolling(5, min_periods=1).mean()
            )
        
        results.append(
            team_matches[["fixture_id"] + [c for c in team_matches.columns if c.startswith(f"{ground}_avg_")]]
        )

    return pd.concat(results)


def compute_standings(df):
    df = df.sort_values("date").reset_index(drop=True)
    standings_data = []

    points = {team: 0 for team in pd.concat([df.home_team, df.away_team]).unique()}
    gf = {team: 0 for team in points}
    ga = {team: 0 for team in points}

    for _, row in df.iterrows():
        table = sorted(points.keys(), key=lambda t: (points[t], gf[t] - ga[t], gf[t]), reverse=True)
        rank = {team: pos+1 for pos, team in enumerate(table)}

        standings_data.append({
            "fixture_id": row.fixture_id,
            "home_rank": rank[row.home_team],
            "away_rank": rank[row.away_team],
            "home_points_before": points[row.home_team],
            "away_points_before": points[row.away_team]
        })
    
        home_pts = 3 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 0
        away_pts = 3 if row.away_goals > row.home_goals else 1 if row.home_goals == row.away_goals else 0

        points[row.home_team] += home_pts
        points[row.away_team] += away_pts
        gf[row.home_team] += row.home_goals
        gf[row.away_team] += row.away_goals
        ga[row.home_team] += row.away_goals
        ga[row.away_team] += row.home_goals

    return pd.DataFrame(standings_data)


if __name__ == "__main__":

    #Load and sort data by date
    df = pd.read_csv("data/epl_matches_partial.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    #Compute
    home_features = compute_team_features(df, "home_team", "home")
    away_features = compute_team_features(df, "away_team", "away")
    standings = compute_standings(df)

    #Droping any fixture_id duplicate
    home_features = home_features.drop_duplicates(subset="fixture_id")
    away_features = away_features.drop_duplicates(subset="fixture_id")
    standings = standings.drop_duplicates(subset="fixture_id")

    # Merge the datasets
    df_final = df.merge(home_features, on="fixture_id", how="left")
    df_final = df_final.merge(away_features, on="fixture_id", how="left")
    df_final = df_final.merge(standings, on="fixture_id", how="left")


    df_final.to_csv("data/epl_sample.csv", index=False)

    print(df["fixture_id"].nunique(), len(df))                # should both be 380
    print(home_features["fixture_id"].nunique(), len(home_features))
    print(away_features["fixture_id"].nunique(), len(away_features))
    print(standings["fixture_id"].nunique(), len(standings))

    print(f"✅ Preprocessed dataset saved with {len(df_final)} rows and {len(df_final.columns)} columns")

