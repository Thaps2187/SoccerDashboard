from dash import Dash, html, dcc
import dash
from dotenv import load_dotenv

load_dotenv() 

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server

def navbar():
    # desired order
    order = ["Overview", "Standings", "Match Predictions", "Players"]

    # sort page_registry according to that order
    pages_sorted = sorted(
        dash.page_registry.values(),
        key=lambda page: order.index(page["name"]) if page["name"] in order else 999
    )


    links = []
    for page in pages_sorted:
        if page.get("path") == "/404":
            continue
        links.append(dcc.Link(page["name"], href=page["path"],
                              style={"marginRight":"16px","textDecoration":"none"}))
    return html.Nav(links, style={"textAlign": "center", "padding":"8px 12px","borderBottom":"2px solid #eee"})

app.layout = html.Div([
    html.H1("⚽ Soccer Analytics Dashboard",
            style= {"textAlign": "center", "color": "darkblue"}),
    navbar(), dash.page_container])

if __name__ == "__main__":
    app.run(debug=True)





# from dash import Dash, html
# import dash

# app = Dash(
#     __name__,
#     use_pages=True,
#     suppress_callback_exceptions=True
# )
# server = app.server  

# app.layout = html.Div([
#     dash.page_container
# ])

# if __name__ == "__main__":
#     app.run(debug=True)
