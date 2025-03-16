import pandas as pd
results_df = pd.read_csv("results.csv")
print(sorted(results_df["home_team"].unique()))