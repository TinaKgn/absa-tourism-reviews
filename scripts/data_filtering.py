"""
Functions for filtering and querying data
"""

import pandas as pd

def classify_tourism_establishment(categories):
    """
    Classify establishment type from categories string for tourism analysis.

    This function identifies only hospitality and food service businesses relevant to
    tourism research using keyword-based matching.

    Keywords:
    - Hotels: hotel, hotels
    - Restaurants: restaurant, restaurants, cafe, cafes (note that bars are not included in this analysis)

    Parameters:
    -----------
    categories : str
        Comma-separated category string from Yelp business data.
        Example: "Mexican, Restaurants, Bars" or "Hotels, Hotels & Travel"

    Returns:
    --------
    str : Classification result
        - 'Hotels': If contains hotel/hotels keywords
        - 'Restaurants': If contains restaurant/restaurants/cafe keywords
        - 'Other': If neither tourism nor food-related
        - 'Unknown': If categories is missing/null

    Notes:
    ------
    - This is a simplified classification for tourism/hospitality analysis
    - Hotels is checked first (priority over Restaurants)
    - Case-insensitive keyword matching
    """
    if pd.isna(categories):
        return 'Unknown'

    cats_lower = str(categories).lower()

    # Check for Hotels & Travel keywords
    hotel_keywords = ['hotel', 'hotels']
    if any(keyword in cats_lower for keyword in hotel_keywords):
        return 'Hotels'

    # Check for Restaurants keywords
    restaurant_keywords = ['restaurant', 'restaurants', 'cafe', 'cafes']
    if any(keyword in cats_lower for keyword in restaurant_keywords):
        return 'Restaurants'

    # Everything else
    return 'Other'


def get_categories_distribution(df_business):
    """
    Get distribution of all categories in business data

    Parameters:
    -----------
    df_business : pd.DataFrame
        DataFrame with business data containing 'categories' column

    Returns:
    --------
    pd.DataFrame : DataFrame with category counts
    """
    from collections import Counter

    all_categories = []
    for categories_str in df_business['categories'].dropna():
        cats = [cat.strip() for cat in str(categories_str).split(',')]
        all_categories.extend(cats)

    category_counts = Counter(all_categories)

    category_df = pd.DataFrame(category_counts.most_common(),
                              columns=['category', 'business_count'])

    return category_df

def filter_by_city_and_establishment_type(df, city, state, establishment_types):
    """
    Filter businesses by city, state, and establishment type(s).

    This function assumes the DataFrame has an 'tourism_establishment_type' column
    created by the classify_tourism_establishment() function.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with business data (must have 'establishment_type' column)
    city : str
        City name (e.g., 'Philadelphia')
    state : str
        State code (e.g., 'PA')
    establishment_types : str or list
        Single establishment type (e.g., 'Restaurants') or
        list of types (e.g., ['Restaurants', 'Hotels & Travel'])
        Valid types: 'Restaurants', 'Hotels & Travel', 'Other', 'Unknown'

    Returns:
    --------
    pd.DataFrame : Filtered DataFrame


    Notes:
    ------
    - Requires 'tourism_establishment_type' column in DataFrame
    - Run classify_tourism_establishment() first to create this column
    - Automatically handles single string or list of strings
    """
    # Check if establishment_type column exists
    if 'tourism_establishment_type' not in df.columns:
        raise ValueError(
            "DataFrame must have 'tourism_establishment_type' column. "
            "Run classify_tourism_establishment() first to create it."
        )

    # Filter by city and state
    df_city = df[(df['city'] == city) & (df['state'] == state)].copy()

    # Convert single establishment type to list for uniform processing
    if isinstance(establishment_types, str):
        establishment_types = [establishment_types]

    # Filter by establishment type
    df_filtered = df_city[
        df_city['tourism_establishment_type'].isin(establishment_types)
    ].copy()

    return df_filtered