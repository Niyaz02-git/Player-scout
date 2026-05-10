import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from processor import (
    CLUSTERING_FEATURES, CLUSTER_NAMES,
    scale_features, save_scaler, load_scaler
)

MODEL_PATH  = "data/kmeans_model.pkl"
SCALER_PATH = "data/scaler.pkl"
DATA_PATH   = "data/players_clustered.csv"


# ── Find optimal K ─────────────────────────────────────────────────────────────

def find_optimal_k(df: pd.DataFrame, k_range=range(2, 11)) -> int:
    X, _ = scale_features(df)
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method — pick where the curve bends")
    ax2.plot(list(k_range), silhouettes, "go-")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — higher is better")

    plt.tight_layout()
    plt.savefig("assets/elbow.png", dpi=150)
    plt.close()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"Best K by silhouette: {best_k} — chart saved to assets/elbow.png")
    return best_k


# ── Train model ────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    X, scaler = scale_features(df)

    model  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    df = df.copy()
    df["cluster"]      = labels
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "pca": pca}, f)

    save_scaler(scaler, SCALER_PATH)
    df.to_csv(DATA_PATH, index=False)
    print(f"Model trained with K={k}. {len(df)} players saved to {DATA_PATH}")
    return df


def cluster_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cluster"] + [f for f in CLUSTERING_FEATURES if f in df.columns]
    return df[cols].groupby("cluster").mean().round(3)


# ── Real-time prediction ───────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_cluster(player_row: dict | pd.Series) -> tuple[int, str]:
    artifacts = load_model()
    scaler    = load_scaler(SCALER_PATH)

    row  = pd.DataFrame([player_row])
    X, _ = scale_features(row, scaler=scaler)

    cluster_id = int(artifacts["model"].predict(X)[0])
    return cluster_id, CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")


def find_similar_players(player_row: dict | pd.Series,
                         n: int = 5,
                         same_cluster_only: bool = True) -> pd.DataFrame:
    artifacts   = load_model()
    scaler      = load_scaler(SCALER_PATH)
    all_players = pd.read_csv(DATA_PATH)

    query_df   = pd.DataFrame([player_row])
    X_query, _ = scale_features(query_df, scaler=scaler)
    X_all, _   = scale_features(all_players, scaler=scaler)

    distances = np.linalg.norm(X_all - X_query, axis=1)
    all_players["_distance"] = distances

    if same_cluster_only:
        cluster_id, _ = predict_cluster(player_row)
        pool = all_players[all_players["cluster"] == cluster_id].copy()
    else:
        pool = all_players.copy()

    pool    = pool[pool["_distance"] > 0.01]
    similar = pool.nsmallest(n, "_distance").drop(columns=["_distance"])
    return similar