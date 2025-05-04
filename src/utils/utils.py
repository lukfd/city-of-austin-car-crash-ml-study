import pandas as pd
import numpy as np
import time
import statsmodels.api as sm

class Timer:
    def __enter__(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the timer and calculates the duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"Executed in {self.duration:.4f} seconds")

class Pickler:
    def __init__(self, filename: str):
        self.filename = filename

    def save(self, obj):
        """Saves an object to a file using pickle."""
        import pickle
        with open(self.filename, 'wb') as f:
            pickle.dump(obj, f)

    def load(self):
        """Loads an object from a file using pickle."""
        import pickle
        with open(self.filename, 'rb') as f:
            return pickle.load(f)

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

def backwards_elimination(X_train, y_train, significance_level=0.05): 
    removed_features = []
    X_opt = X_train.copy()
    X_opt = sm.add_constant(X_opt)  # Add constant, don't cast to int

    while True: 
        obj_OLS = sm.OLS(y_train, X_opt).fit()  # <--- y first, X second
        p_values = obj_OLS.pvalues
        max_p_value = p_values.max()

        if max_p_value > significance_level:
            feature_to_remove = p_values.idxmax()
            if feature_to_remove == 'const':
                print('Constant term has high p-value. Stopping.')
                break
            removed_features.append(feature_to_remove)
            X_opt.drop(columns=[feature_to_remove], inplace=True)
        else:
            break
    return X_opt, removed_features

def print_linear_regression_scores(model: str, y, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    y_mean = np.mean(y)
    rmse_percentage = (rmse / y_mean) * 100

    print(f"{model} Scores:")
    print(f"MAE: {mae:.4f} MSE: {mse:.4f} RMSE: {rmse:.4f} RMSE%: {rmse_percentage:.2f}% R^2: {r2:.4f}")
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "RMSE%": rmse_percentage,
        "R^2": r2
    }
