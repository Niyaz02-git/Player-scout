#importing libraries 

import time                     
import requests                 
import pandas as pd            
from io import StringIO         
from bs4 import BeautifulSoup   


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


FBREF_URLS = {
    "Premier League": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "La Liga":        "https://fbref.com/en/comps/12/stats/La-Liga-Stats",
    "Bundesliga":     "https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
    "Serie A":        "https://fbref.com/en/comps/11/stats/Serie-A-Stats",
    "Ligue 1":        "https://fbref.com/en/comps/13/stats/Ligue-1-Stats",
}


def scrape_league_stats(league: str = "Premier League") -> pd.DataFrame:

    url = FBREF_URLS.get(league)
    if not url:
        raise ValueError(f"Unknown league: {league}. Choose from {list(FBREF_URLS.keys())}")

    print(f"[FBref] Fetching {league} stats...")

    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()  # raises error if page didn't load successfully

    soup = BeautifulSoup(res.text, "lxml")

    table = soup.find("table", {"id": "stats_standard"})

    if table is None:
        import re
        comments = soup.find_all(string=lambda t: isinstance(t, str) and "stats_standard" in t)
        if comments:
            table_html = re.search(
                r'(<table[^>]*id="stats_standard".*?</table>)',
                comments[0],
                re.DOTALL
            )
            if table_html:
                table = BeautifulSoup(table_html.group(1), "lxml").find("table")

    if table is None:
        raise RuntimeError("[FBref] Could not find stats table. FBref may have changed their HTML.")

    df = pd.read_html(StringIO(str(table)), header=1)[0]

    df = df[df["Player"] != "Player"].copy()
    df = df.dropna(subset=["Player"])

    rename_map = {
        "Player": "player",
        "Nation": "nation",
        "Pos":    "position",
        "Squad":  "team",
        "Age":    "age",
        "MP":     "matches_played",
        "Min":    "minutes",
        "Gls":    "goals",
        "Ast":    "assists",
        "xG":     "xg",
        "xAG":    "xAG",
        "PrgC":   "progressive_carries",  
        "PrgP":   "progressive_passes",    
        "PrgR":   "progressive_receptions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_cols = ["matches_played", "minutes", "goals", "assists",
                    "xg", "xag", "progressive_carries",
                    "progressive_passes", "progressive_receptions"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") 

 
    df["league"] = league

    time.sleep(4)

    print(f"[FBref] Got {len(df)} players for {league}")
    return df


def scrape_all_leagues(save_path: str = "data/fbref_all.csv") -> pd.DataFrame:
 
    frames = []

    for league in FBREF_URLS:
        try:
            frames.append(scrape_league_stats(league))
            time.sleep(5)  # extra delay between leagues to be safe
        except Exception as e:
            print(f"[FBref] Failed {league}: {e}")
          
    df = pd.concat(frames, ignore_index=True)

    
    df.to_csv(save_path, index=False)
    print(f"[FBref] Saved {len(df)} total players to {save_path}")
    return df


def search_fbref_player(name: str, df: pd.DataFrame) -> pd.Series | None:
 
    
    mask = df["player"].str.contains(name, case=False, na=False)
    results = df[mask]

    if results.empty:
        return None

   
    return results.sort_values("minutes", ascending=False).iloc[0]