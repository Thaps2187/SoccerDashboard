import dash
from dash import html


dash.register_page(__name__, path="/players", name="Players")

layout = html.Div([
    html.H1("Player Stats"),
    html.P('Dashboard still under construction')
    ])
