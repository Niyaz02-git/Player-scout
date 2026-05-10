import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

CLUSTER_COLORS = {
    "Prolific Striker":   "#E85D24",
    "Creative Playmaker": "#7F77DD",
    "Box-to-Box":         "#1D9E75",
    "Wide Attacker":      "#EF9F27",
    "Defensive Anchor":   "#378ADD",
}

RADAR_METRICS = [
    "goals_per90", "assists_per90", "xg_per90",
    "xag_per90", "progressive_carries_per90", "progressive_passes_per90",
]

RADAR_LABELS = [
    "Goals / 90", "Assists / 90", "xG / 90",
    "xA / 90", "Prog. Carries / 90", "Prog. Passes / 90",
]


# ── Radar chart ────────────────────────────────────────────────────────────────

def radar_chart(player_row: dict | pd.Series, title: str = "") -> go.Figure:
    values = [float(player_row.get(m, 0) or 0) for m in RADAR_METRICS]
    values_closed = values + [values[0]]
    labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(127, 119, 221, 0.2)",
        line=dict(color="#7F77DD", width=2),
        name=title,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.5, font=dict(size=13)),
    )
    return fig


# ── Cluster scatter plot ───────────────────────────────────────────────────────

def cluster_scatter(df: pd.DataFrame, highlight_player: str | None = None) -> go.Figure:
    fig = px.scatter(
        df,
        x="pc1", y="pc2",
        color="cluster_name",
        hover_data={
            "player": True, "team": True,
            "goals_per90": ":.2f", "assists_per90": ":.2f",
            "pc1": False, "pc2": False,
        },
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.7,
        labels={"pc1": "PC 1", "pc2": "PC 2", "cluster_name": "Playing style"},
    )

    if highlight_player and highlight_player in df["player"].values:
        row = df[df["player"] == highlight_player].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row["pc1"]], y=[row["pc2"]],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="white",
                        line=dict(color="#E85D24", width=2)),
            text=[highlight_player],
            textposition="top center",
            textfont=dict(size=11),
            showlegend=False,
            name=highlight_player,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title="Playing style", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
    )
    return fig


# ── Rating timeline ────────────────────────────────────────────────────────────

def rating_timeline(fixtures: list[dict]) -> go.Figure:
    dates, ratings, opponents = [], [], []

    for f in fixtures:
        stat   = f.get("statistics", [{}])[0] if f.get("statistics") else {}
        rating = stat.get("games", {}).get("rating")
        if rating:
            dates.append(f.get("fixture", {}).get("date", "")[:10])
            ratings.append(float(rating))
            opponents.append(f.get("teams", {}).get("away", {}).get("name", ""))

    if not dates:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=ratings,
        mode="lines+markers",
        line=dict(color="#7F77DD", width=2),
        marker=dict(size=6, color="#7F77DD"),
        hovertext=opponents,
        hovertemplate="<b>Rating: %{y}</b><br>vs %{hovertext}<extra></extra>",
    ))
    fig.add_hline(y=7.0, line_dash="dot",
                  line_color="rgba(128,128,128,0.4)",
                  annotation_text="Avg (7.0)")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(range=[4, 10], showgrid=True,
                   gridcolor="rgba(128,128,128,0.1)", title="Rating"),
        margin=dict(l=20, r=20, t=10, b=20),
    )
    return fig


# ── Top players bar chart ──────────────────────────────────────────────────────

def top_players_bar(df: pd.DataFrame, metric: str = "goals_per90",
                    n: int = 15, cluster_filter: str | None = None) -> go.Figure:
    filtered = df[df["cluster_name"] == cluster_filter] if cluster_filter else df
    top      = filtered.nlargest(n, metric)[["player", "team", "cluster_name", metric]].copy()
    key      = cluster_filter or ""
    color    = CLUSTER_COLORS.get(key, "#7F77DD")

    fig = go.Figure(go.Bar(
        x=top[metric].round(3),
        y=top["player"],
        orientation="h",
        marker_color=color,
        text=top[metric].round(2),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + metric.replace("_", " ") + ": %{x}<extra></extra>",
    ))

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        margin=dict(l=20, r=60, t=10, b=20),
    )
    return fig


# ── Similar players table ──────────────────────────────────────────────────────

def similar_players_table(similar_df: pd.DataFrame) -> go.Figure:
    cols_to_show   = ["player", "team", "cluster_name",
                      "goals_per90", "assists_per90", "rating"]
    cols_available = [c for c in cols_to_show if c in similar_df.columns]
    display        = similar_df[cols_available].copy()

    fig = go.Figure(go.Table(
        header=dict(
            values=["Player", "Team", "Style",
                    "Goals/90", "Assists/90", "Rating"][:len(cols_available)],
            fill_color="#1a1a2e",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[display[c] for c in cols_available],
            fill_color=[["rgba(127,119,221,0.05)", "rgba(0,0,0,0)"] *
                        (len(display) // 2 + 1)][:len(display)],
            font=dict(size=11),
            align="left",
            format=["", "", "", ".3f", ".3f", ".1f"][:len(cols_available)],
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig