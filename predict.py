import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load Historical Data
results_df = pd.read_csv("results.csv")
goals_df = pd.read_csv("goalscorers.csv")
spi_intl_df = pd.read_csv("spi_global_rankings_intl.csv")
results_df["date"] = pd.to_datetime(results_df["date"])
goals_df["date"] = pd.to_datetime(goals_df["date"])

# Load Trained Model and Preprocessors
model = joblib.load("football_prediction_model.pkl")
imputer = joblib.load("imputer.pkl")
team_encoder = joblib.load("team_encoder.pkl")
tournament_encoder = joblib.load("tournament_encoder.pkl")

# Define 2026 Teams with Correct Names from results.csv
teams_2026 = [
    "Brazil", "France", "Argentina", "England", "Spain", "Germany", "Italy", "Netherlands",
    "Portugal", "Belgium", "Uruguay", "Croatia", "Mexico", "Colombia", "United States", "Canada",
    "Morocco", "Senegal", "Japan", "South Korea", "Australia", "Qatar", "Saudi Arabia", "Iran",
    "Egypt", "Nigeria", "Ghana", "Algeria", "Tunisia", "Cameroon", "Ivory Coast", "Mali",
    "Chile", "Peru", "Ecuador", "Paraguay", "Denmark", "Sweden", "Poland", "Switzerland",
    "Serbia", "Ukraine", "Turkey", "Greece", "Norway", "Wales", "Scotland", "Republic of Ireland"
]
assert len(teams_2026) == 48, "Need exactly 48 teams"

# Extend Encoders
all_teams = pd.concat([results_df["home_team"], results_df["away_team"]]).unique()
unseen_teams = [team for team in teams_2026 if team not in all_teams]
if unseen_teams:
    print(f"Warning: These teams are not in results.csv: {unseen_teams}")
    all_teams = sorted(set(all_teams).union(teams_2026))
    team_encoder.fit(all_teams)
    print("Team Encoder retrained.")

all_tournaments = results_df["tournament"].unique()
future_tournaments = ["FIFA World Cup"]
unseen_tournaments = [t for t in future_tournaments if t not in all_tournaments]
if unseen_tournaments:
    print(f"Warning: These tournaments are not in results.csv: {unseen_tournaments}")
    all_tournaments = sorted(set(all_tournaments).union(future_tournaments))
    tournament_encoder.fit(all_tournaments)
    print("Tournament Encoder retrained.")

# Simulate 2026 Group Stage
match_data = []
start_date = datetime(2026, 6, 11)
for group in range(16):
    group_teams = teams_2026[group * 3:(group + 1) * 3]
    match_data.append({"date": start_date, "home_team": group_teams[0], "away_team": group_teams[1], "tournament": "FIFA World Cup", "neutral": True})
    match_data.append({"date": start_date, "home_team": group_teams[1], "away_team": group_teams[2], "tournament": "FIFA World Cup", "neutral": True})
    match_data.append({"date": start_date, "home_team": group_teams[2], "away_team": group_teams[0], "tournament": "FIFA World Cup", "neutral": True})

future_df = pd.DataFrame(match_data)
future_df["date"] = pd.to_datetime(future_df["date"])

# Form Feature
def precompute_form(df, historical_df, window=5):
    combined_df = pd.concat([historical_df, df], ignore_index=True).sort_values("date")
    teams = pd.concat([combined_df["home_team"], combined_df["away_team"]]).unique()
    for team in teams:
        mask = (combined_df["home_team"] == team) | (combined_df["away_team"] == team)
        team_matches = combined_df[mask].copy()
        if "match_outcome" in team_matches.columns:
            points = np.where(
                (team_matches["home_team"] == team) & (team_matches["match_outcome"] == 1) |
                (team_matches["away_team"] == team) & (team_matches["match_outcome"] == 0), 3,
                np.where(team_matches["match_outcome"] == 2, 1, 0))
            form = pd.Series(points, index=team_matches.index).rolling(window, min_periods=1).mean().shift(1).fillna(0)
        else:
            form = pd.Series(0, index=team_matches.index)
        combined_df.loc[mask & (combined_df["home_team"] == team), "home_form"] = form
        combined_df.loc[mask & (combined_df["away_team"] == team), "away_form"] = form
    return combined_df.tail(len(df)).fillna(0)

future_df = precompute_form(future_df, results_df)

# Tournament Weight
tournament_weights = {"FIFA World Cup": 1.0, "Friendly": 0.3, "Qualifiers": 0.7}
future_df["tournament_weight"] = future_df["tournament"].map(tournament_weights).fillna(0.5)

# Static SPI
spi_intl_dict = spi_intl_df.set_index("name")[["off", "def"]].to_dict()
future_df["home_spi_offense"] = future_df["home_team"].map(spi_intl_dict["off"]).fillna(1.5)
future_df["away_spi_offense"] = future_df["away_team"].map(spi_intl_dict["off"]).fillna(1.5)
future_df["home_spi_defense"] = future_df["home_team"].map(spi_intl_dict["def"]).fillna(1.5)
future_df["away_spi_defense"] = future_df["away_team"].map(spi_intl_dict["def"]).fillna(1.5)
future_df["spi_offense_diff"] = future_df["home_spi_offense"] - future_df["away_spi_defense"]

# Goalscorers Features
avg_goals = goals_df.groupby("team")["minute"].count() / goals_df.groupby("team").apply(
    lambda x: len(x[["date", "home_team", "away_team"]].drop_duplicates()), include_groups=False)
future_df["home_avg_goals"] = future_df["home_team"].map(avg_goals).fillna(0)
future_df["away_avg_goals"] = future_df["away_team"].map(avg_goals).fillna(0)

# Encoding
future_df["home_team_encoded"] = team_encoder.transform(future_df["home_team"])
future_df["away_team_encoded"] = team_encoder.transform(future_df["away_team"])
future_df["tournament_encoded"] = tournament_encoder.transform(future_df["tournament"])

# Features
features = ["home_team_encoded", "away_team_encoded", "tournament_encoded", "neutral", 
            "home_form", "away_form", "tournament_weight", "home_avg_goals", "away_avg_goals",
            "home_spi_offense", "away_spi_defense", "spi_offense_diff"]
X_future = future_df[features]

# Impute NaNs
X_future = pd.DataFrame(imputer.transform(X_future), columns=features)

# Predict
predictions = model.predict(X_future)
future_df["predicted_outcome"] = predictions
outcome_map = {0: "Away Win", 1: "Home Win", 2: "Draw"}
future_df["predicted_result"] = future_df["predicted_outcome"].map(outcome_map)

# Display Results
print("FIFA World Cup 2026 Group Stage Predictions:")
for i, row in future_df.iterrows():
    print(f"{row['date'].date()}: {row['home_team']} vs {row['away_team']} - {row['predicted_result']}")

# Save Predictions
future_df.to_csv("fifa_2026_predictions.csv", index=False)
print("Predictions saved to fifa_2026_predictions.csv")