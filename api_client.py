#importing necessary libraries 

import os
import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": os.getenv("API_FOOTBALL_KEY", "")}

#League names mapped to their api-football ids
LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1:": 61,
    "Champions league": 2,
}

# Base Request Function

def _get(endpoint: str, params: dict)-> dict:
    try:
        res = requests.get(
            f"{BASE_URL}/{endpoint}",

            headers=HEADERS,

            params=params,

            timeout=10
        )

        res.raise_for_status()
        return res.json()
    
    except requests.RequestException as e:
        print(f"[API Error]{e}")
        return {}
    
# Player Search

def search_players(name: str, season:int= 2024)  -> list[dict]:
    if len(name) < 3:
        return[]
  
    data = _get("players", {"search": name, "season": season}) # pyright: ignore[reportUndefinedVariable]
    return data.get("response",[])

def get_player_by_id(player_id: int, season: int = 2024) -> dict | None:
    data = _get("players", {"id": player_id, "season": season}) # pyright: ignore[reportUndefinedVariable]
    response = data.get("response", [])
    return response[0] if response else None
 
# League stats

def get_top_scorers(league_id: int = 39, season: int = 2024) -> list[dict]:
    data = _get("players/topscorers", {"league": league_id, "season": season}) # pyright: ignore[reportUndefinedVariable]
    return data.get("response", [])

def get_top_assists(league_id: int = 39, season: int = 2024) -> list[dict]:
    data = _get("players/topassists", {"league": league_id, "season": season}) # pyright: ignore[reportUndefinedVariable]
    return data.get("response", [])

def get_top_clean_sheets(league_id: int = 39, season: int = 2024) -> list[dict]:
    all_keepers = []
    page        = 1
    while True:
        data = _get("players", { # pyright: ignore[reportUndefinedVariable]
            "league":   league_id,
            "season":   season,
            "position": "Goalkeeper",  # filter to goalkeepers only
            "page":     page
        })
 
        response = data.get("response", [])
 
        if not response:
            break
 
        all_keepers.extend(response)
 
        total_pages = data.get("paging", {}).get("total", 1)
        if page >= total_pages:
            break
 
        page += 1
 
    for keeper in all_keepers:
        stats = keeper.get("statistics", [{}])[0] if keeper.get("statistics") else {}
        cs = stats.get("goals", {}).get("conceded", 0)
        # API-Football stores clean sheets under goalkeeper-specific field
        keeper["clean_sheets"] = stats.get("games", {}).get("lineups", 0) - cs \
            if cs is not None else 0
       
    all_keepers.sort(
        key=lambda x: x.get("clean_sheets", 0),
        reverse=True
    )
 
    return all_keepers
 