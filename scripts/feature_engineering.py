"""
Functions for creating engineered features
"""

import pandas as pd


def add_engagement_features(df):
    """
    Add engagement-based features from useful/funny/cool votes on Yelp

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'useful', 'funny', 'cool' columns

    Returns:
    --------
    pd.DataFrame : DataFrame with new engagement features added
    """
    df = df.copy()

    # High quality review (useful votes number >= 10)
    df['high_quality_review'] = (df['useful'] >= 10).astype(int)

    # Has any engagement
    df['has_engagement'] = (
        (df['useful'] > 0) |
        (df['funny'] > 0) |
        (df['cool'] > 0)
    ).astype(int)

    # Total engagement score
    df['total_votes'] = df['useful'] + df['funny'] + df['cool']

    # Engagement type (categorical)
    def categorize_engagement(row):
        if row['useful'] + row['funny'] + row['cool'] == 0:
            return 'No Engagement'
        elif row['useful'] > row['funny'] and row['useful'] > row['cool']:
            return 'Primarily Useful'
        elif row['funny'] > row['useful'] and row['funny'] > row['cool']:
            return 'Primarily Funny'
        elif row['cool'] > row['useful'] and row['cool'] > row['funny']:
            return 'Primarily Cool'
        else:
            return 'Mixed Engagement'

    df['engagement_type'] = df.apply(categorize_engagement, axis=1)

    # Engagement level (ordinal)
    def engagement_level(total_votes):
        if total_votes == 0:
            return 'None'
        elif total_votes <= 3:
            return 'Low'
        elif total_votes <= 9:
            return 'Medium'
        else:
            return 'High'

    df['engagement_level'] = df['total_votes'].apply(engagement_level)

    print("✓ Added engagement features:")
    print("  - high_quality_review")
    print("  - has_engagement")
    print("  - total_votes")
    print("  - engagement_type")
    print("  - engagement_level")

    return df


def add_seasons(df, date_column='review_date'):
    """
    Add seasons from date column

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    date_column : str
        Name of date column (default: 'review_date')

    Returns:
    --------
    pd.DataFrame : DataFrame with season features added
    """
    df = df.copy()

    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter

    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'

    df['season'] = df['month'].apply(get_season)

    print("✓ Added season features:")
    print("  - year, month, quarter")
    print("  - day_of_week, season")

    return df