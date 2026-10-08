"""Freature engineering for CNP transaction fraud model"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0    

def haversine_km(lat1,lon1, lat2, lon2) -> pd.Series:
    """Vectorised haversine distance in km between two points (lat1, lon1) and (lat2, lon2) """
    lat1, lon1, lat2,lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def add_geo_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Distance between Cardholder and home location and merchant location"""
    df = df.copy()
    df["geo_distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour of day, day of week, and timesincelasttrasaction per card"""
    df = df.copy()
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    #Time since this card previoud transaction in minutes
    
    prev_time = df.groupby("cc_num")["unix_time"].shift(1)
    df["mins_since_last_txn"] = (df["unix_time"] - prev_time) / 60
    df["mins_since_last_txn"] = df["mins_since_last_txn"].fillna(24*60)  
    return df.sort_index()

def add_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """This is standardization of the transaction amount per cardholder"""
    df = df.copy()
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    grouped = df.groupby("cc_num")["amt"]
    expanding_mean = grouped.apply(lambda s: s.shift(1).expanding().mean())
    expanding_std = grouped.apply(lambda s: s.shift(1).expanding().std())

    expanding_mean = expanding_mean.reset_index(level=0, drop=True)
    expanding_std = expanding_std.reset_index(level=0, drop=True)

    df["amt_hist_mean"] = expanding_mean
    df["amt_hist_std"] = expanding_std.replace(0, np.nan)  # Avoid division by zero
    df["amt_zscore"] = (df["amt"] - df["amt_hist_mean"]) / df["amt_hist_std"]
    
    #First transaction per card (or zero-variance history) has no baseline

    df["amt_zscore"] = df["amt_zscore"].fillna(0)
    df["amt_hist_mean"] = df["amt_hist_mean"].fillna(df["amt"])
    return df.sort_index()

def fit_category_risk_encooding(train_df: pd.DataFrame) -> pd.Series:
    """encoding categorial data"""
    return train_df.groupby("category")["is_fraud"].mean()

def apply_category_risk_encoding(df: pd.DataFrame, category_encoding: pd.Series) -> pd.DataFrame:
    df = df.copy()
    global_rate = category_encoding.mean()
    df["category_fraud_rate"] = (df["category"].map(category_encoding).fillna(global_rate))
    return df

def build_features(df: pd.DataFrame, category_encoding: pd.Series = None) -> pd.DataFrame:
    """Master Orchestrator"""
    df = add_geo_velocity(df)
    df = add_time_features(df)
    df = add_amount_zscore(df)

    if category_encoding is None:
        category_encoding = fit_category_risk_encooding(df)
    df = apply_category_risk_encoding(df, category_encoding)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df, category_encoding

FEATURE_COLUMNS = [
    "amt",
    "geo_distance_km",
    "hour",
    "day_of_week",
    "mins_since_last_txn",
    "amt_zscore",
    "category_fraud_rate",
    "city_pop",
]

DROP_COLUMNS = [
    "cc_num",
    "trans_num",
    "first",
    "last",
    "street",
    "job",
    "merchant",
    "dob",
]
