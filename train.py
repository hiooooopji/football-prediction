import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load Historical Datasets
results_df = pd.read_csv("results.csv")
goals_df = pd.read_csv("goalscorers.csv")
spi_intl_df = pd.read_csv("spi_global_rankings_intl.csv")

# Convert dates
results_df["date"] = pd.to_datetime(results_df["date"])
goals_df["date"] = pd.to_datetime(goals_df["date"])

# Match Outcome
results_df["match_outcome"] = results_df.apply(
    lambda row: 1 if row["home_score"] > row["away_score"] else (0 if row["home_score"] < row["away_score"] else 2), axis=1)

# Form Feature
def precompute_form(df, window=5):
    df = df.sort_values("date")
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    for team in teams:
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        team_matches = df[mask].copy()
        points = np.where(
            (team_matches["home_team"] == team) & (team_matches["match_outcome"] == 1) |
            (team_matches["away_team"] == team) & (team_matches["match_outcome"] == 0), 3,
            np.where(team_matches["match_outcome"] == 2, 1, 0))
        form = pd.Series(points, index=team_matches.index).rolling(window, min_periods=1).mean().shift(1).fillna(0)
        df.loc[mask & (df["home_team"] == team), "home_form"] = form
        df.loc[mask & (df["away_team"] == team), "away_form"] = form
    return df.fillna(0)

results_df = precompute_form(results_df)

# Tournament Weight
tournament_weights = {"FIFA World Cup": 1.0, "Friendly": 0.3, "Qualifiers": 0.7}
results_df["tournament_weight"] = results_df["tournament"].map(tournament_weights).fillna(0.5)

# Static SPI
spi_intl_dict = spi_intl_df.set_index("name")[["off", "def"]].to_dict()
results_df["home_spi_offense"] = results_df["home_team"].map(spi_intl_dict["off"]).fillna(1.5)
results_df["away_spi_offense"] = results_df["away_team"].map(spi_intl_dict["off"]).fillna(1.5)
results_df["home_spi_defense"] = results_df["home_team"].map(spi_intl_dict["def"]).fillna(1.5)
results_df["away_spi_defense"] = results_df["away_team"].map(spi_intl_dict["def"]).fillna(1.5)
results_df["spi_offense_diff"] = results_df["home_spi_offense"] - results_df["away_spi_defense"]

# Goalscorers Features
avg_goals = goals_df.groupby("team")["minute"].count() / goals_df.groupby("team").apply(
    lambda x: len(x[["date", "home_team", "away_team"]].drop_duplicates()), include_groups=False)
results_df["home_avg_goals"] = results_df["home_team"].map(avg_goals).fillna(0)
results_df["away_avg_goals"] = results_df["away_team"].map(avg_goals).fillna(0)

# Encoding with Separate Encoders
team_encoder = LabelEncoder()
tournament_encoder = LabelEncoder()
all_teams = pd.concat([results_df["home_team"], results_df["away_team"]]).unique()
team_encoder.fit(all_teams)
results_df["home_team_encoded"] = team_encoder.transform(results_df["home_team"])
results_df["away_team_encoded"] = team_encoder.transform(results_df["away_team"])
results_df["tournament_encoded"] = tournament_encoder.fit_transform(results_df["tournament"])

# Features and Target
features = ["home_team_encoded", "away_team_encoded", "tournament_encoded", "neutral", 
            "home_form", "away_form", "tournament_weight", "home_avg_goals", "away_avg_goals",
            "home_spi_offense", "away_spi_defense", "spi_offense_diff"]
target = "match_outcome"
X = results_df[features]
y = results_df[target]

# Impute NaNs
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split and SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy={0: int(y_train.value_counts()[1]), 2: int(y_train.value_counts()[1])}, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Model
model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, 
    colsample_bytree=0.9, objective="multi:softprob", num_class=3, 
    random_state=42, n_jobs=-1, min_child_weight=2
)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win", "Draw"]))

# Save Model and Preprocessors
joblib.dump(model, "football_prediction_model.pkl")
joblib.dump(team_encoder, "team_encoder.pkl")
joblib.dump(tournament_encoder, "tournament_encoder.pkl")
joblib.dump(imputer, "imputer.pkl")
print("Model and preprocessors saved!")