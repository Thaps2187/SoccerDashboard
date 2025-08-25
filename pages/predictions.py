import dash
from dash import html


dash.register_page(__name__, path="/predictions", name="Match Predictions")

layout = html.Div([
    html.H1("Match Predictions"),
    html.P('Model still under training')
    ])
