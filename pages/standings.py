import dash
from dash import html, dcc, Input, Output, dash_table, State, get_asset_url
import pandas as pd
import plotly.express as px
import os, re
from pathlib import Path


dash.register_page(__name__, path="/standings", name="Standings")

BASE = Path(__file__).resolve().parent.parent   # go up from pages/ to repo root
DATA = BASE / "data"

LEAGUE_PREFIX = {
    "EPL": "epl",
    "LL": "ll",
    "SA": "sa",
    "BL": "bl",
    "L1": "l1",
}

def opt(img, value, title):
    return {
        "label": html.Span(
            [
                html.Img(
                    src=get_asset_url(f"logos/{img}"),  # <-- URL Dash serves
                    height=40,
                    style={"marginRight": 12}
                ),
            ],
            style={"display": "inline-flex", "alignItems": "center"},
            title=title
        ),
        "value": value,
    }

def split_home_away(series):
    """Convert a Series with Home_ / Away_ stats into a clean DataFrame"""
    data = []
    for ground in ["Home", "Away"]:
        row = {
            "ground": ground,
            "win": series[f"{ground}_wins"],
            "draw": series[f"{ground}_draw"],
            "lose": series[f"{ground}_lose"],
            "scores": series[f"{ground}_goals_for"],
            "conceded": series[f"{ground}_goals_against"]
        }
        data.append(row)
    return pd.DataFrame(data)

layout = html.Div([
    dcc.RadioItems(
                id="league-radio",
                options=[
                    opt("epl.png",        "EPL", "Premier League"),
                    opt("laliga.png",     "LL",  "La Liga"),
                    opt("seriea.png",     "SA",  "Serie A"),
                    opt("bundesliga.png", "BL",  "Bundesliga"),
                    opt("ligue1.png",     "L1",  "Ligue 1"),
                ],
                value="EPL",
                labelStyle={"display": "inline-block", "marginRight": "20px"},
                inputStyle={"marginRight": "8px"}  # circle + logo sit side-by-side
                ),
    
    # Season dropdown (populated dynamically from selected league)
    dcc.Dropdown(id="season-dropdown", placeholder="Select season", style={"width":"240px"}),

    html.P("Click on a team to see more of it's performance"),

    # Outputs
    dash_table.DataTable(
        id="standings-table",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"padding":"6px","textAlign":"center"},
        style_header={"fontWeight":"bold"},
        style_data_conditional=[{
            "if": {"column_id": "team"},
            "textDecoration": "underline",
            "fontWeight": "600",
        }]
    ),

    dcc.Store(id="results-store"),

    # Team charts shown after click
    html.Div(id="team-graph", style={"marginTop": 16}),

    dcc.Graph(id="standings-graph"),


],  style={"padding":"16px"}
)

# Callback 1: Update season options when league changes
@dash.callback(
    Output("season-dropdown", "options"),
    Output("season-dropdown", "value"),
    Input("league-radio", "value"),
)
def update_season_dropdown(league):
    seasons = ["2010/11", "2011/12", "2012/13", "2013/14", "2014/15", "2015/16",
               "2016/17", "2017/18", "2018/19", "2019/20", "2020/21", "2021/22",
               "2022/23", "2023/24", "2024/25"]
    opts = [{"label": s, "value": s} for s in seasons]
    # default to most recent season for that league
    value = seasons[-1] if seasons else None
    return opts, value


# -- Callback 2: render standings for league+season
@dash.callback(
        Output("standings-graph", "figure"),
        Output("standings-table", "data"),
        Output("standings-table", "columns"),
        Input("league-radio", "value"),
        Input("season-dropdown", "value"),
)
def render_standings(league, season):
    # Loading in the csv league data
    yr_selected = int(re.findall(r'\b\d+\b', season)[0])
    prefix = LEAGUE_PREFIX[league]
    csv_path = DATA / f"{prefix}_tables" / f"{prefix}_table_{yr_selected}.csv"

    if not csv_path.exists():
        return html.Div(
            [
                html.H4("Data not found", style={"color": "crimson"}),
                html.P(f"Could not find: {csv_path}")
            ]
    )


    lg_table = pd.read_csv(csv_path)
    d = lg_table.sort_values("Rank")

    # Table: show rank, team, pts, results, gf, ga, gd, form, qualification, 
    cols = ["Rank", "Team", "MP", "W","D", "L", "GF", "GA", "Pts", "Form", "Qualification"]
    data = d[cols].to_dict("records")
    columns = [{"name": c.upper(), "id": c} for c in cols]

    # The figure
    # Bar chart: points by team (ordered by rank)
    fig = px.bar(
        d, x="Team", y="GD",color= "Qualification",
        hover_data=["GF","GA","GD","Rank"],
        title=f"Standings — {league} {season} (Points)"
    )
    return fig, data, columns

# ---- CLICK HANDLER: click a team → plot a team chart
@dash.callback(
    Output("team-graph", "children"),
    Input("standings-table", "active_cell"),
    Input("league-radio", "value"),
    Input("season-dropdown", "value"),
    State("standings-table", "data"),
    prevent_initial_call=True
)
def on_cell_click(active_cell, league, season, rows):
    # Only respond if the user clicked the TEAM column
    if not active_cell or active_cell.get("column_id") != "Team":
        raise dash.exceptions.PreventUpdate
    i = active_cell["row"]
    team = rows[i]["Team"]

    yr_selected = int(re.findall(r'\b\d+\b', season)[0])
    prefix = LEAGUE_PREFIX[league]
    csv_path = DATA / f"{prefix}_tables" / f"{prefix}_table_{yr_selected}.csv"

    if not csv_path.exists():
        return html.Div(
            [
                html.H4("Data not found", style={"color": "crimson"}),
                html.P(f"Could not find: {csv_path}")
            ]
    )

    df = pd.read_csv(csv_path)
    team_index = df[df["Team"] == team].index[0]

    df_col = df.columns
    df_col = df_col[13:]
    results_data = df[df_col]
    results_data = results_data.iloc[team_index]

    ha = split_home_away(results_data)

    df_results = ha.melt(id_vars="ground", value_vars=["win","draw","lose"],
                                  var_name="result", value_name="count")
    
    df_goals = ha.melt(id_vars="ground", value_vars=["scores", "conceded"], 
                       var_name="type", value_name="goals")


    # This creates a bar graph to demostrate the Team's performance both home and away
    fig_goals = px.bar(data_frame= df_goals, 
                       x= "type", 
                       y= "goals", 
                       color= "ground", 
                       title= f"{team}'s {season} home and away goals")
    
    fig_results = px.bar(data_frame= df_results, 
                       x="result", 
                       y="count", 
                       color="ground", 
                       barmode="group",
                       title= f"{team}'s {season} home and away performance")

    # ← You now have the team string. Call your own data source next.
    return html.Div(
        [
            html.H4(f"{team} ({league}, {season})"),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_results), style={"flex": "1 1 380px", "minWidth": 320}),
                    html.Div(dcc.Graph(figure=fig_goals),   style={"flex": "1 1 380px", "minWidth": 320}),
                ],
                style={"display": "flex", "gap": 16, "flexWrap": "wrap", "alignItems": "stretch"}
            ),
        ],
        style={"paddingTop": 8}
    )

