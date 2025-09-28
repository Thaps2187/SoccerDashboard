import dash
from dash import html, Input, Output, dcc, get_asset_url
from dash.exceptions import PreventUpdate
from pathlib import Path
from sqlalchemy import text
import pandas as pd
import plotly.graph_objects as go
import json
from src.db import engine 


dash.register_page(__name__, path="/predictions", name="Match Predictions")


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
        "value": title,
    }

def make_h2h_table(homes, scores, aways, title=None) -> go.Figure:
    def parse_score(s):
        h, a = s.split(":"); return int(h), int(a)

    home_fill, away_fill, score_fill, home_font, away_font = [], [], [], [], []
    for s in scores:
        h, a = parse_score(s)
        if h > a:  # home win
            home_fill.append("#d1fae5"); away_fill.append("#fee2e2")
            home_font.append("#065f46"); away_font.append("#991b1b")
        elif h < a:  # away win
            home_fill.append("#fee2e2"); away_fill.append("#d1fae5")
            home_font.append("#991b1b"); away_font.append("#065f46")
        else:  # draw
            home_fill.append("#e5e7eb"); away_fill.append("#e5e7eb")
            home_font.append("#374151"); away_font.append("#374151")
        score_fill.append("#f9fafb")

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Home","Score","Away"],
                    fill_color="#111827", font=dict(color="white"), align="left", height=26),
        cells=dict(
            values=[homes, scores, aways],
            fill_color=[home_fill, score_fill, away_fill],   # column-wise color lists
            font=dict(color=[home_font, ["#111827"]*len(scores), away_font]),
            align="left", height=26
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140)
    fig.update_layout(
        title_text=title or "Head-to-Head (Last 5)",
        title_x=0.5,
        title_font_size=18,
        margin=dict(l=0, r=0, t=60, b=0),
        height=140
    )
    return fig

def no_h2h_fig(game):
    fig = go.Figure()
    fig.add_annotation(text=f"No head-to-head yet: {game}",
                       x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0))
    return fig

def empty_fig(msg=""):
    f = go.Figure()
    if msg:
        f.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    f.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    return f

CURRENT_SEASON = 2025

layout = html.Div([
    html.H1("Match Predictions"),
    dcc.RadioItems(
                id="league-radio",
                options=[
                    opt("epl.png",        "EPL", "Premier League"),
                    opt("laliga.png",     "LL",  "La Liga"),
                    opt("seriea.png",     "SA",  "Serie A"),
                    opt("bundesliga.png", "BL",  "Bundesliga"),
                    opt("ligue1.png",     "L1",  "Ligue 1"),
                ],
                value="Premier League",
                labelStyle={"display": "inline-block", "marginRight": "20px"},
                inputStyle={"marginRight": "8px"}  # circle + logo sit side-by-side
                ),
    # Match dropdown (populated dynamically from upcoming games)
    dcc.Dropdown(id= "Match-dropdown", placeholder="Select Match", style={"width":"240px"}),

    dcc.Store(id="matches-store", data=[]),

    # Predicted match results
    dcc.Graph(id= "predicted-results", figure=empty_fig()),

    # Head-to-Head 5 last outcomes between the 2 teams
    dcc.Graph(id= "h2h-graph", config={"displayModeBar": False})
])

#Updates the dropdown menu for the upcoming matches
@dash.callback(
    Output("Match-dropdown", "options"),
    Output("Match-dropdown", "value"),
    Output("matches-store", "data"),
    Input("league-radio", "value"),
)
def update_match_dropdown(league):

    league_id = engine.connect().execute(
        text("SELECT league_id FROM leagues WHERE name = :name"), {"name": league}).scalar()
    
    fixture_sql = text("""
                    SELECT
                        m.fixture_id,
                        m.home_team_id, ht.name AS home_team,
                        m.away_team_id, at.name  AS away_team,
                        (ht.name || ' vs ' || at.name) AS game
                    FROM matches AS m
                    JOIN teams AS ht ON ht.team_id = m.home_team_id
                    JOIN teams AS at ON at.team_id = m.away_team_id
                    WHERE m.league_id = :league_id AND m.status = 'NS' AND m.season_year = 2025
                    ORDER BY m.match_date ASC 
                    LIMIT 10;
                """)
    
    upcoming_fx = engine.connect().execute(fixture_sql, {"league_id": league_id}).fetchall()
    upcoming_fx = pd.DataFrame(upcoming_fx)

    games = upcoming_fx["game"].tolist()

    opts = [{"label": m, "value": m} for m in games]
    # default to most recent season for that league
    value = games[0] if games else None

    records = upcoming_fx.to_dict("records")   

    return opts, value, records

# Updates the predicted results for the 2 teams
@dash.callback(
    Output("predicted-results", "figure"),
    Input("Match-dropdown", "value"),
    Input("matches-store", "data"), 
    prevent_initial_call=True
)
def render_pred(game, matches):
    if not matches or not game:    
        raise PreventUpdate
    
    matches = pd.DataFrame.from_records(matches)

    fixture_id = matches.loc[matches["game"].str.casefold().eq(game.casefold()), "fixture_id"].item()
    home_tm = matches.loc[matches["game"].str.casefold().eq(game.casefold()), "home_team"].item()
    away_tm = matches.loc[matches["game"].str.casefold().eq(game.casefold()), "away_team"].item()

    pred_sql = text("""
                    SELECT p_home, p_draw, p_away
                    FROM fixture_predictions
                    WHERE fixture_id = :fixture_id;""")

    pred_results = engine.connect().execute(pred_sql, {"fixture_id": fixture_id}).fetchone()

    if pred_results is None:
        return empty_fig(f"No prediction for {game}")


    pred_results = list(pred_results)
    pred_results = [round(v * 100) for v in pred_results]

    if len(pred_results) != 3:
        return empty_fig("Bad prediction shape")
    
    fig = go.Figure(go.Pie(
    labels=[home_tm, "Draw", away_tm], values=pred_results,
    hole=0.55, sort=False, direction="clockwise",
    texttemplate="%{label}<br>%{value}%", hovertemplate="%{label}: %{value}%<extra></extra>"
    ))
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
    fig.update_layout(
    title_text=f"Predicted Outcome: {home_tm} vs {away_tm}",
    title_x=0.5,                
    title_font_size=18,
    margin=dict(t=60, b=10, l=10, r=10), 
    showlegend=False,
    )


    return fig

# Updates the last 5 Head-to-Head between the 2 teams
@dash.callback(
    Output("h2h-graph", "figure"),
    Input("matches-store", "data"),
    Input("Match-dropdown", "value")
)
def update_h2h(matches, game):
    if not matches or not game:    
        raise PreventUpdate
    
    matches = pd.DataFrame.from_records(matches)

    first_tm = matches.loc[matches["game"].str.casefold().eq(game.casefold()), "home_team_id"].item()
    second_tm = matches.loc[matches["game"].str.casefold().eq(game.casefold()), "away_team_id"].item()

    h2h_sql = text("""
                SELECT 
                    ht.name AS home_team,
                    at.name  AS away_team,
                    (CAST(m.home_goals AS TEXT) || ':' || CAST(m.away_goals AS TEXT)) AS score
                FROM matches AS m
                JOIN teams AS ht ON ht.team_id = m.home_team_id
                JOIN teams AS at ON at.team_id = m.away_team_id
                WHERE m.status = 'FT' AND 
                (((m.home_team_id = :first_team AND m.away_team_id = :second_team)) 
                    OR (m.home_team_id = :second_team AND m.away_team_id = :first_team))
                ORDER BY m.match_date DESC 
                LIMIT 5;
            """)
    

    h2h = engine.connect().execute(h2h_sql, {"first_team": first_tm, "second_team": second_tm}).fetchall()
    h2h = pd.DataFrame(h2h)

    if h2h.empty:
        return no_h2h_fig(game)

    homes  = h2h["home_team"]
    scores = h2h["score"]
    aways  = h2h["away_team"]

    return make_h2h_table(homes, scores, aways, title=f"Head-to-Head (Last 5): {game}")

    
