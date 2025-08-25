import pandas as pd

fixtures = pd.read_csv("data/epl_fixtures_2024.csv")
stats = pd.read_csv("data/epl_stats.csv")

# Merge fixtures with stats collected so far
merged = fixtures.merge(stats, on="fixture_id", how="left")

# Save as a working merged file
merged.to_csv("data/epl_matches_partial.csv", index=False)

print(f"✅ Updated dataset saved with {len(merged)} rows ({merged['home_corners'].notna().sum()} with stats)")

