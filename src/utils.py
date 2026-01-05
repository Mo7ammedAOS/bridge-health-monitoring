"""
Utility functions for Bridge Digital Twin ML Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from src.config import *


def load_data(filepath):
    """
    Load bridge digital twin data and parse Timestamps
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        df: Pandas DataFrame with parsed Timestamps
    """
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df


def handle_missing_values(df, method='ffill'):
    """
    Handle missing values in time series data
    
    Args:
        df: Input DataFrame
        method: 'ffill' (forward fill) or 'bfill' (backward fill)
    
    Returns:
        df: DataFrame with missing values handled
    """
    if method == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'bfill':
        df = df.fillna(method='bfill').fillna(method='ffill')
    return df


def remove_outliers(df, columns, n_sigma=3):
    """
    Remove outliers using n-sigma rule (IQR-based clipping)
    
    Args:
        df: Input DataFrame
        columns: List of column names to process
        n_sigma: Number of IQR ranges for clipping
    
    Returns:
        df: DataFrame with outliers clipped
    """
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - n_sigma * IQR
        upper = Q3 + n_sigma * IQR
        df_clean[col] = df_clean[col].clip(lower, upper)
    
    return df_clean


def time_series_split(X, y, train_ratio=0.8):
    """
    Split time series data without shuffling
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_ratio: Proportion of data for training
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(train_ratio * len(X))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    
    return metrics


def plot_feature_importance(importance_df, top_n=10, save_path=None):
    """
    Plot feature importance bar chart
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def save_model(model, filename, directory=MODELS_DIR):
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filename: Name of file (e.g., 'xgb_model.pkl')
        directory: Directory to save to
    """
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"✓ Model saved: {filepath}")


def load_model(filename, directory=MODELS_DIR):
    """
    Load trained model from disk
    
    Args:
        filename: Name of file
        directory: Directory to load from
    
    Returns:
        model: Loaded model object
    """
    filepath = os.path.join(directory, filename)
    model = joblib.load(filepath)
    print(f"✓ Model loaded: {filepath}")
    return model


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_metrics(metrics, model_name):
    """Print model metrics in formatted way"""
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric:10s}: {value:.4f}")
    print("-" * 40)