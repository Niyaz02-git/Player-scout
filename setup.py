import os
import pandas as pd

from api_client    import get_top_scorers, get_top_assists, LEAGUES
from fbref_scraper import scrape_all_leagues
from processor     import (build_api_dataframe, engineer_features,
                           merge_fbref, scale_features)
from cluster       import find_optimal_k, train_model, cluster_means

os.makedirs("data",   exist_ok=True)
os.makedirs("assets", exist_ok=True)


# ── Build dataset ──────────────────────────────────────────────────────────────

def build_dataset():
    print("Step 1: Scraping FBref...")
    fbref_df = scrape_all_leagues(save_path="data/fbref_all.csv")

    print("\nStep 2: Fetching API-Football top players...")
    api_rows = []
    for league_name, league_id in LEAGUES.items():
        if league_name == "Champions League":
            continue
        print(f"  {league_name}")
        api_rows += get_top_scorers(league_id=league_id,  season=2024)
        api_rows += get_top_assists(league_id=league_id, season=2024)

    api_df = build_api_dataframe(api_rows).drop_duplicates(subset=["player_id"])
    print(f"  {len(api_df)} unique players from API")

    print("\nStep 3: Merging + engineering features...")
    merged_df = merge_fbref(api_df, fbref_df)
    merged_df = engineer_features(merged_df)
    merged_df = merged_df[merged_df["minutes"] >= 300]
    merged_df = merged_df.dropna(subset=["goals_per90"])
    print(f"  Final dataset: {len(merged_df)} players")

    return merged_df


# ── Train model ────────────────────────────────────────────────────────────────

def run():
    df = build_dataset()

    print("\nStep 4: Finding optimal K...")
    best_k     = find_optimal_k(df)
    user_input = input(f"Enter K to use [{best_k}]: ").strip()
    k          = int(user_input) if user_input else best_k

    print(f"\nTraining KMeans with K={k}...")
    clustered_df = train_model(df, k=k)

    print("\nStep 5: Cluster summary — use this to name your clusters in processor.py")
    print(cluster_means(clustered_df).to_string())
    print("\nDone. Run:  python dashboard.py")


if __name__ == "__main__":
    run()