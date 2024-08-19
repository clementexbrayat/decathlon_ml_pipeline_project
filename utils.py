import pandas as pd

def preprocess_calendar_features(df) :
    try:
        df["day_id"] = pd.to_datetime(df["day_id"])
        df["day_id_week"] = df.day_id.dt.isocalendar().week
        df["day_id_month"] = df["day_id"].dt.month
        df["day_id_year"] = df["day_id"].dt.year
        return df
    except Exception as e:
        print(f"Error preprocessing calendar features: {e}")
        raise