import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from api_client    import search_players
from processor     import build_api_dataframe, engineer_features, merge_fbref
from cluster       import predict_cluster, find_similar_players
from charts        import (radar_chart, cluster_scatter,
                           top_players_bar, similar_players_table)

# ── App init ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Player Scout",
    suppress_callback_exceptions=True,
)

try:
    ALL_PLAYERS = pd.read_csv("data/players_clustered.csv")
except FileNotFoundError:
    ALL_PLAYERS = pd.DataFrame()
    print("[Warning] Run setup.py first.")

try:
    FBREF_DF = pd.read_csv("data/fbref_all.csv")
except FileNotFoundError:
    FBREF_DF = pd.DataFrame()


# ── Layout ─────────────────────────────────────────────────────────────────────

SEARCH_BAR = dbc.Row([
    dbc.Col(
        dcc.Input(
            id="search-input",
            type="text",
            placeholder="Search any player...  e.g. Salah, Mbappé, Pedri",
            debounce=True,
            style={"width": "100%", "fontSize": "16px", "padding": "12px 16px",
                   "borderRadius": "8px", "border": "1px solid #333",
                   "background": "#1a1a2e", "color": "white"},
        ),
        width=8,
    ),
    dbc.Col(
        dcc.Dropdown(
            id="season-dropdown",
            options=[{"label": f"{s}/{str(s+1)[2:]}", "value": s}
                     for s in range(2020, 2025)],
            value=2024,
            clearable=False,
        ),
        width=2,
    ),
    dbc.Col(
        html.Div(id="request-counter",
                 style={"color": "#666", "fontSize": "12px", "paddingTop": "14px"}),
        width=2,
    ),
], className="mb-4")

PLAYER_CARD = html.Div(
    id="player-card",
    children=[
        html.Div("Search for a player above to see their full profile.",
                 style={"color": "#666", "textAlign": "center",
                        "padding": "60px 0", "fontSize": "15px"}),
    ]
)

TABS = dcc.Tabs(id="main-tabs", value="tab-search", children=[

    # ── Tab 1: Player search ───────────────────────────────────────────────────
    dcc.Tab(label="Player Search", value="tab-search", children=[
        html.Div([SEARCH_BAR, PLAYER_CARD], style={"padding": "24px"}),
    ]),

    # ── Tab 2: League explorer ─────────────────────────────────────────────────
    dcc.Tab(label="League Explorer", value="tab-explorer", children=[
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="cluster-filter",
                    options=[{"label": n, "value": n} for n in [
                        "All", "Prolific Striker", "Creative Playmaker",
                        "Box-to-Box", "Wide Attacker", "Defensive Anchor",
                    ]],
                    value="All", clearable=False,
                ), width=4),
                dbc.Col(dcc.Dropdown(
                    id="metric-filter",
                    options=[
                        {"label": "Goals per 90",   "value": "goals_per90"},
                        {"label": "Assists per 90", "value": "assists_per90"},
                        {"label": "xG per 90",      "value": "xg_per90"},
                        {"label": "Rating",         "value": "rating"},
                    ],
                    value="goals_per90", clearable=False,
                ), width=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cluster-scatter"), width=7),
                dbc.Col(dcc.Graph(id="top-players-bar"), width=5),
            ]),
        ], style={"padding": "24px"}),
    ]),
])

app.layout = html.Div([
    html.Div([
        html.H1("Player Scout",
                style={"fontSize": "22px", "fontWeight": "500",
                       "margin": "0", "color": "white"}),
        html.Span("Powered by API-Football + FBref + KMeans",
                  style={"fontSize": "12px", "color": "#666", "marginLeft": "12px"}),
    ], style={"padding": "16px 24px", "borderBottom": "1px solid #222",
              "display": "flex", "alignItems": "center"}),
    TABS,
    dcc.Store(id="current-player-store"),
], style={"minHeight": "100vh", "background": "#0d0d1a", "fontFamily": "sans-serif"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("player-card", "children"),
    Output("current-player-store", "data"),
    Input("search-input", "value"),
    Input("season-dropdown", "value"),
    prevent_initial_call=True,
)
def search_and_display(name, season):
    if not name or len(name.strip()) < 3:
        return html.Div("Type at least 3 characters to search.",
                        style={"color": "#666", "textAlign": "center", "padding": "40px"}), {}

    results = search_players(name.strip(), season=season)

    if not results:
        return html.Div(f'No players found for "{name}".',
                        style={"color": "#E85D24", "textAlign": "center", "padding": "40px"}), {}

    api_df = build_api_dataframe(results[:1])

    if not FBREF_DF.empty:
        api_df = merge_fbref(api_df, FBREF_DF)

    api_df      = engineer_features(api_df)
    player      = api_df.iloc[0]
    player_dict = player.to_dict()

    try:
        cluster_id, cluster_name = predict_cluster(player_dict)
        similar_df = find_similar_players(player_dict, n=5)
    except Exception as e:
        print(f"[Cluster error] {e}")
        cluster_name = "Unknown"
        similar_df   = pd.DataFrame()

    card = build_profile_card(results[0]["player"], player, cluster_name, similar_df)
    return card, player_dict


def build_profile_card(api_player, player, cluster_name, similar_df):
    photo_url  = api_player.get("photo", "")
    stat_items = [
        ("Goals",        player.get("goals", 0)),
        ("Assists",      player.get("assists", 0)),
        ("Matches",      player.get("matches_played", 0)),
        ("Minutes",      player.get("minutes", 0)),
        ("Rating",       f"{player.get('rating', 0):.1f}"),
        ("Goals / 90",   f"{player.get('goals_per90', 0):.2f}"),
        ("Assists / 90", f"{player.get('assists_per90', 0):.2f}"),
        ("xG / 90",      f"{player.get('xg_per90', 0):.2f}"),
    ]

    return html.Div([

        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Img(src=photo_url,
                             style={"width": "80px", "borderRadius": "50%",
                                    "border": "2px solid #333"}), width="auto"),
            dbc.Col([
                html.H2(api_player.get("name", ""),
                        style={"fontSize": "22px", "fontWeight": "500",
                               "margin": "0", "color": "white"}),
                html.Div([
                    html.Span(player.get("team", ""),
                              style={"color": "#aaa", "fontSize": "14px"}),
                    html.Span(" · ", style={"color": "#444"}),
                    html.Span(player.get("position", ""),
                              style={"color": "#aaa", "fontSize": "14px"}),
                    html.Span(" · ", style={"color": "#444"}),
                    html.Span(cluster_name,
                              style={"background": "#7F77DD22", "color": "#7F77DD",
                                     "padding": "2px 10px", "borderRadius": "12px",
                                     "fontSize": "12px", "border": "1px solid #7F77DD55"}),
                ], style={"marginTop": "6px"}),
            ]),
        ], align="center", className="mb-4"),

        # ── Stat grid ─────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div(str(val), style={"fontSize": "22px",
                                              "fontWeight": "500", "color": "white"}),
                    html.Div(label, style={"fontSize": "11px", "color": "#666",
                                           "marginTop": "2px"}),
                ], style={"background": "#111122", "borderRadius": "8px",
                          "padding": "14px 16px", "border": "1px solid #222"}),
                width=3, className="mb-3"
            )
            for label, val in stat_items
        ], className="mb-4"),

        # ── Radar + similar players ────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H6("Stat profile",
                        style={"color": "#aaa", "fontSize": "12px",
                               "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "8px"}),
                dcc.Graph(figure=radar_chart(player.to_dict(), ""),
                          style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], width=5),
            dbc.Col([
                html.H6("Similar players — same playing style",
                        style={"color": "#aaa", "fontSize": "12px",
                               "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "8px"}),
                dcc.Graph(figure=similar_players_table(similar_df)
                          if not similar_df.empty else go.Figure(),
                          style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], width=7),
        ]),
    ])


@app.callback(
    Output("cluster-scatter", "figure"),
    Output("top-players-bar",  "figure"),
    Input("cluster-filter",   "value"),
    Input("metric-filter",    "value"),
)
def update_explorer(cluster, metric):
    if ALL_PLAYERS.empty:
        empty = go.Figure()
        return empty, empty

    cluster_val = None if cluster == "All" else cluster
    scatter     = cluster_scatter(ALL_PLAYERS)
    bar         = top_players_bar(ALL_PLAYERS, metric=metric,
                                  cluster_filter=cluster_val, n=15)
    return scatter, bar


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)