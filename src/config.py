"""
Configuration file for Bridge Digital Twin ML Project
"""

import os

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# FEATURES
# ============================================
STRUCTURAL_FEATURES = [
    'Strain_microstrain',
    'Deflection_mm',
    'Vibration_ms2',
    'Modal_Frequency_Hz',
    'Fatigue_Accumulation_au'
]

ENVIRONMENTAL_FEATURES = [
    'Temperature_C',
    'Humidity_percent',
    'Wind_Speed_ms'
]

TRAFFIC_FEATURES = [
    'Traffic_Volume_vph',
    'Vehicle_Load_tons'
]

# Combined feature set for modeling
FEATURE_COLS = STRUCTURAL_FEATURES + ENVIRONMENTAL_FEATURES + TRAFFIC_FEATURES

# Target variable
TARGET_COL = 'Structural_Health_Index_SHI'

# Features to AVOID (data leakage)
EXCLUDE_FEATURES = [
    'SHI_Predicted_24h',
    'SHI_Predicted_7d',
    'SHI_Predicted_30d',
    'Probability_of_Failure_PoF',
    'Maintenance_Alert'
]

# ============================================
# MODEL HYPERPARAMETERS
# ============================================

# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================
# TRAIN/TEST SPLIT
# ============================================
TRAIN_TEST_SPLIT_RATIO = 0.8

# ============================================
# DEGRADATION ANALYSIS
# ============================================
DEGRADATION_WINDOW_SIZE = 30  # minutes
DEGRADATION_THRESHOLDS = {
    'stable': 0.0005,      # abs(rate) < 0.0005
    'fast_degrade': -0.0015  # rate < -0.0015
}

# ============================================
# SURVIVAL ANALYSIS
# ============================================
SHI_THRESHOLD_EVENT = 0.6  # Health index threshold for maintenance event
RISK_TIME_HORIZONS = [30, 60, 90]  # days for risk prediction

# ============================================
# VISUALIZATION
# ============================================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
COLOR_PALETTE = 'Set2'