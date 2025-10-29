import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import requests
import json

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Predictive Maintenance — Live Telemetry"),
    dcc.Interval(id="tick", interval=1000, n_intervals=0),
    dcc.Graph(id="telemetry"),
    html.Div(id="prob", style={"fontWeight":"bold"}),
    dcc.Store(id="series", data=[]),
], style={"maxWidth": "900px", "margin":"auto"})

@app.callback(
    Output("series", "data"),
    Input("tick", "n_intervals"),
    State("series", "data"),
)
def generate(n, series):
    # synthetic streaming row
    rng = np.random.default_rng()
    row = [
        float(70 + 5*np.sin(n/10) + rng.normal(0, 0.6)),
        float(1 + 0.3*np.sin(n/8) + rng.normal(0, 0.05)),
        float(10 + 2*np.sin(n/12) + rng.normal(0, 0.4)),
        float(500 + 40*np.sin(n/15) + rng.normal(0, 5))
    ]
    series = (series or []) + [row]
    return series[-200:]

@app.callback(
    Output("telemetry", "figure"),
    Output("prob", "children"),
    Input("series", "data"),
)
def plot(series):
    arr = np.array(series) if series else np.zeros((1,4))
    fig = go.Figure()
    for i, name in enumerate(["temp","vib","curr","rpm"]):
        fig.add_trace(go.Scatter(y=arr[:,i], name=name))
    prob_txt = "Failure Probability: (start API at :8000 and call /predict)"
    try:
        if len(arr) >= 24:
            resp = requests.post("http://localhost:8000/predict", json={"series": arr.tolist()}, timeout=0.5)
            if resp.ok:
                prob = resp.json()["failure_probability"]
                prob_txt = f"Failure Probability: {prob:.3f}"
    except Exception:
        pass
    fig.update_layout(legend_orientation="h", margin=dict(l=20,r=20,t=20,b=20))
    return fig, prob_txt

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
