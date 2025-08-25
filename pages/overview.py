import dash
from dash import html, dcc

overview_output = '''
## ⚽ Soccer Analytics Dashboard  

Welcome to the **Soccer Analytics Dashboard**!  

This interactive tool lets you dive into the beautiful game through data-driven insights and visualizations.  

### What you can explore:
- 📊 **League Standings** – compare team performance across seasons  
- 🏟️ **Team Tracking** – follow your favorite clubs and their trends  
- 👤 **Player Stats** – view individual player metrics  
- 🔮 **Match Predictions** – see data-driven probabilities for upcoming games  

Whether you're a **stats enthusiast** or just a **curious fan**, this dashboard makes exploring soccer simple, engaging, and fun.
'''

dash.register_page(__name__, path="/", name="Overview")

layout = html.Div([
    dcc.Markdown(overview_output, style={'whiteSpace': 'preline'})
    ])
