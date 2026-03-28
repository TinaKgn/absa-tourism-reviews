"""
Functions for data visualization
"""

import pandas as pd
import matplotlib.pyplot as plt



def print_dataset_summary(df, target_years=None):
    """
    Print summary statistics for the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The complete dataset containing at least 'year' and 'business_id'.

    target_years : list or None, optional
        • If a list is provided, only those years are summarized.
        • If None (default), summaries are generated for **all years present**
          in the DataFrame (sorted).
    """
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    # If no target years provided → use all years in dataset
    if target_years is None:
        target_years = sorted(df["year"].unique())
        print(f"Summarizing ALL YEARS in dataset: {target_years}")

    for year in target_years:
        df_year = df[df["year"] == year]

        print(f"\n{year}:")
        print(f"  Total reviews: {len(df_year):,}")

        # Establishment types optional
        if "establishment_type" in df.columns:
            print(f"  Restaurants: {len(df_year[df_year['establishment_type'] == 'Restaurants']):,}")
            print(f"  Hotels: {len(df_year[df_year['establishment_type'] == 'Hotels & Travel']):,}")

        print(f"  Unique businesses: {df_year['business_id'].nunique()}")
        print(f"  Unique users: {df_year['user_id'].nunique()}")

        if "review_stars" in df.columns:
            print(f"  Avg review rating: {df_year['review_stars'].mean():.2f}")

        # Seasonal breakdown
        if "season" in df.columns:
            seasonal = df_year.groupby("season").size()
            print("  Seasonal distribution:")
            for season in ["Winter", "Spring", "Summer", "Fall"]:
                count = seasonal.get(season, 0)
                pct = (count / len(df_year) * 100) if len(df_year) > 0 else 0
                print(f"    {season:8s}: {count:4,} ({pct:5.1f}%)")