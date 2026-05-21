# PlayerScout — Football Player Intelligence Dashboard

A search-first football analytics dashboard that clusters players by playing style using KMeans machine learning and surfaces similar players in real time.

Built with Python, Plotly Dash, Scikit-learn, and API-Football.

---

## What it does

- **Search any player** from the dataset and instantly see their full stat profile
- **Playing style classification** — KMeans clusters players into groups like Pure Striker, Complete Attacker, Deep Playmaker, Defensive Midfielder
- **Similar players** — finds the 5 most stylistically similar players using Euclidean distance in feature space
- **Radar chart** — visual stat profile showing goals, assists, shot accuracy, key passes, dribbles
- **League Explorer** — scatter plot of all 170+ players colored by cluster, top performers bar chart with dropdown filters

---

## Tech stack

| Layer | Tool |
|---|---|
| Data collection | API-Football (free tier) |
| Data processing | Python, Pandas |
| Machine learning | Scikit-learn KMeans + PCA |
| Dashboard | Plotly Dash + Dash Bootstrap Components |
| Visualizations | Plotly |

---

## Project structure

```
player_scout/
├── api_client.py        # API-Football wrapper
├── fbref_scraper.py     # FBref scraper for advanced stats
├── processor.py         # Data cleaning + feature engineering
├── cluster.py           # KMeans training + real-time prediction
├── charts.py            # All Plotly chart builders
├── setup.py             # One-time dataset builder + model trainer
├── dashboard.py         # Main Dash app
├── data/
│   ├── players_clustered.csv   # Full dataset with cluster labels
│   ├── kmeans_model.pkl        # Trained KMeans + PCA model
│   └── scaler.pkl              # Fitted StandardScaler
├── assets/
│   └── elbow.png               # K selection chart
├── .env                        # API key (never commit this)
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Get a free API key
Sign up at [api-sports.io](https://dashboard.api-football.com/register).
Free tier gives 100 requests per day.

### 2. Clone the repo
```bash
git clone https://github.com/Niyaz02-git/Player-scout.git
cd Player-scout
```

### 3. Create virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Add your API key
Create a `.env` file in the root folder:
```
API_FOOTBALL_KEY=your_key_here
```

### 6. Build the dataset and train the model
```bash
python setup.py
```
This fetches top scorers and assisters from API-Football across 5 leagues, engineers per-90 features, runs the elbow method, trains KMeans, and saves everything to the `data/` folder.

When prompted, open `assets/elbow.png` and pick K (recommended: 4 or 5).

After training, the script prints cluster means. Use these to name your clusters in `processor.py` under `CLUSTER_NAMES`.

### 7. Launch the dashboard
```bash
python dashboard.py
```
Open your browser at `http://127.0.0.1:8050`

---

## How search works

```
User types player name
        ↓
Search local CSV (data/players_clustered.csv)
        ↓                    ← 0 API requests
Match by name (case-insensitive, partial match)
        ↓
predict_cluster()   ← KMeans model loaded from disk
        ↓
find_similar_players()   ← Euclidean distance in scaled feature space
        ↓
Render: stat grid + radar chart + similar players table
```

No API requests are used during search — everything runs locally.

---

## Features used for clustering

| Feature | Description |
|---|---|
| goals_per90 | Goals scored per 90 minutes |
| assists_per90 | Assists per 90 minutes |
| yellow_card_rate | Yellow cards per game |
| key_passes | Passes leading to a shot |
| shot_accuracy | Shots on target / total shots |
| pass_accuracy | Pass completion percentage |
| dribbles_success | Successful dribbles |

---

## Limitations

- Dataset is limited to top scorers and assisters from API-Football free tier (~170 players across 5 leagues)
- FBref advanced stats (xG, xA, progressive passes) unavailable due to scraping restrictions — xG/xA columns show 0
- Player search only works for players in the dataset — upgrading to API-Football Basic plan unlocks full player search

---

## Refreshing data

To fetch fresh data and retrain the model:
```bash
python setup.py
```

---

## Author

Mohammed Niyaz Ali
[github.com/Niyaz02-git](https://github.com/Niyaz02-git)
