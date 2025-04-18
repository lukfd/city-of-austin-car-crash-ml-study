import pandas as pd
import numpy as np

def extract_timestamp(timestamps: pd.Series) -> pd.DataFrame:
    if not isinstance(timestamps, pd.Series):
        raise ValueError("Input must be a Pandas Series.")
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("Input Series must contain datetime objects.")

    extracted_features = pd.DataFrame(index=timestamps.index)
    extracted_features['hour'] = timestamps.dt.hour
    extracted_features['day_of_week'] = timestamps.dt.dayofweek
    extracted_features['month'] = timestamps.dt.month
    extracted_features['year'] = timestamps.dt.year
    extracted_features['day_of_month'] = timestamps.dt.day
    extracted_features['weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
    extracted_features['hour_sin'] = np.sin(2 * np.pi * extracted_features['hour'] / 24)
    extracted_features['hour_cos'] = np.cos(2 * np.pi * extracted_features['hour'] / 24)
    extracted_features['month_sin'] = np.sin(2 * np.pi * extracted_features['month'] / 12)
    extracted_features['month_cos'] = np.cos(2 * np.pi * extracted_features['month'] / 12)
    return extracted_features