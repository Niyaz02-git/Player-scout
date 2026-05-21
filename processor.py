#importing libraries

import numpy as np              
import pandas as pd           
from sklearn.preprocessing import StandardScaler   
import pickle                   


# ── Clustering configuration ───────────────────────────────────────────────────
CLUSTERING_FEATURES = [
    "goals_per90",
    "assists_per90",
    "yellow_card_rate",
    "key_passes",
    "shot_accuracy",
    "pass_accuracy",
    "dribbles_success",
]
CLUSTER_NAMES = {
    0: "Prolific Striker",
    1: "Creative Playmaker",
    2: "Box-to-Box",
    3: "Wide Attacker",
    4: "Defensive Anchor",
}


# ── Parsing API-Football responses ─────────────────────────────────────────────

def parse_api_player(response_item: dict) -> dict:
    p    = response_item["player"]                                          # player bio info
    stat = response_item["statistics"][0] if response_item.get("statistics") else {}  # season stats
    goals    = stat.get("goals",    {}) or {}
    passes   = stat.get("passes",   {}) or {}
    dribbles = stat.get("dribbles", {}) or {}
    cards    = stat.get("cards",    {}) or {}
    games    = stat.get("games",    {}) or {}
    shots    = stat.get("shots",    {}) or {}

    minutes = games.get("minutes") or 0  # default to 0 if missing
    return {
        "player_id":        p.get("id"),
        "player":           p.get("name"),
        "age":              p.get("age"),
        "nationality":      p.get("nationality"),
        "photo":            p.get("photo"),           # URL to player headshot image
        "team":             stat.get("team",   {}).get("name"),
        "league":           stat.get("league", {}).get("name"),
        "position":         games.get("position"),    # e.g. 'Attacker', 'Midfielder'
        "matches_played":   games.get("appearences") or 0,
        "minutes":          minutes,
        "goals":            goals.get("total")    or 0,
        "assists":          goals.get("assists")  or 0,
        "shots_total":      shots.get("total")    or 0,
        "shots_on":         shots.get("on")       or 0,   # shots on target
        "passes_total":     passes.get("total")   or 0,
        "key_passes":       passes.get("key")     or 0,   # passes leading to a shot
        "pass_accuracy":    passes.get("accuracy") or 0,  # percentage of passes completed
        "dribbles_success": dribbles.get("success") or 0,
        "yellow_cards":     cards.get("yellow")   or 0,
        "red_cards":        cards.get("red")      or 0,
        "rating":           float(games.get("rating") or 0),  # average match rating
    }


def build_api_dataframe(api_responses: list[dict]) -> pd.DataFrame:
    rows = [parse_api_player(r) for r in api_responses]  
    return pd.DataFrame(rows)                           


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  

    # Replace 0 minutes with NaN to avoid division by zero
    # We'll fill the resulting NaN values with 0 afterward
    minutes = df["minutes"].replace(0, np.nan)

    # Basic per-90 stats from API-Football
    df["goals_per90"]   = (df["goals"]   / minutes * 90).fillna(0)
    df["assists_per90"] = (df["assists"] / minutes * 90).fillna(0)

    # Advanced per-90 stats from FBref
    df["xg_per90"]  = (df.get("xg",  0) / minutes * 90).fillna(0)
    df["xag_per90"] = (df.get("xag", 0) / minutes * 90).fillna(0)

    # Progressive action rates from FBref

    df["progressive_carries_per90"] = (
        df.get("progressive_carries", pd.Series(0, index=df.index)) / minutes * 90
    ).fillna(0)

    df["progressive_passes_per90"] = (
        df.get("progressive_passes", pd.Series(0, index=df.index)) / minutes * 90
    ).fillna(0)

    # Yellow card rate = cards per game
    df["yellow_card_rate"] = (
        df["yellow_cards"] / df["matches_played"].replace(0, np.nan)
    ).fillna(0)

    # Bonus derived stats
    df["goal_contributions_per90"] = df["goals_per90"] + df["assists_per90"]
    df["shot_accuracy"] = (
        df["shots_on"] / df["shots_total"].replace(0, np.nan)
    ).fillna(0)

    return df


def merge_fbref(api_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:

    api_df   = api_df.copy()
    fbref_df = fbref_df.copy()

    api_df["_key"]   = api_df["player"].str.lower().str.strip()
    fbref_df["_key"] = fbref_df["player"].str.lower().str.strip()

    advanced_cols = ["_key", "xg", "xag", "progressive_carries",
                     "progressive_passes", "progressive_receptions"]
    available = [c for c in advanced_cols if c in fbref_df.columns]

    merged = api_df.merge(fbref_df[available], on="_key", how="left")
    merged.drop(columns=["_key"], inplace=True)  
    return merged


# ── Feature scaling ────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame, scaler=None):
    available = [f for f in CLUSTERING_FEATURES if f in df.columns]
    X = df[available].fillna(0).values  

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)   
    else:
        X_scaled = scaler.transform(X)       

    return X_scaled, scaler


def save_scaler(scaler, path: str = "data/scaler.pkl"):

    with open(path, "wb") as f:
        pickle.dump(scaler, f) 


def load_scaler(path: str = "data/scaler.pkl"):

    with open(path, "rb") as f:
        return pickle.load(f)  