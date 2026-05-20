"""
preprocessing.py
----------------
Utility functions for cleaning, encoding, and scaling the
Diabetes Health Indicators dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os


# ──────────────────────────────────────────────
# 1. LOADING
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# 2. INSPECTION HELPERS
# ──────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Print a structured overview of the DataFrame."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"\n📐 Shape            : {df.shape}")
    print(f"\n🔢 Data Types:\n{df.dtypes}")
    print(f"\n📊 Descriptive Stats:\n{df.describe()}")
    print(f"\n❓ Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n🔁 Duplicate Rows   : {df.duplicated().sum()}")
    print("=" * 60)


# ──────────────────────────────────────────────
# 3. MISSING VALUE HANDLING
# ──────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame,
                           strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values.

    Parameters
    ----------
    df       : Input DataFrame
    strategy : 'mean', 'median', or 'most_frequent'

    Returns
    -------
    DataFrame with no missing values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    print(f"✅ Missing values handled using '{strategy}' strategy.")
    return df


# ──────────────────────────────────────────────
# 4. DUPLICATE REMOVAL
# ──────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and reset the index."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"✅ Removed {removed} duplicate row(s). Remaining: {len(df)}")
    return df


# ──────────────────────────────────────────────
# 5. ENCODING
# ──────────────────────────────────────────────

def label_encode_column(df: pd.DataFrame,
                         column: str) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Apply Label Encoding to a single column.

    Returns the modified DataFrame and the fitted LabelEncoder
    (needed to inverse-transform predictions later).
    """
    df = df.copy()
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    print(f"✅ Label-encoded column: '{column}'  →  classes: {list(le.classes_)}")
    return df, le


def one_hot_encode_columns(df: pd.DataFrame,
                            columns: list[str]) -> pd.DataFrame:
    """Apply One-Hot Encoding to the specified columns."""
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    print(f"✅ One-Hot encoded columns: {columns}")
    return df


# ──────────────────────────────────────────────
# 6. FEATURE SCALING
# ──────────────────────────────────────────────

def standardize_features(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          columns: list[str] | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on X_train and transform both splits.

    Parameters
    ----------
    X_train  : Training features
    X_test   : Testing features
    columns  : Columns to scale; if None, all numeric columns are used.

    Returns
    -------
    (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    if columns is None:
        columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])

    print(f"✅ Standardized {len(columns)} feature(s).")
    return X_train, X_test, scaler


# ──────────────────────────────────────────────
# 7. PREPARE TARGETS
# ──────────────────────────────────────────────

def prepare_targets(df: pd.DataFrame) -> dict:
    """
    Extract the three target columns from the DataFrame.

    Returns
    -------
    {
        'binary'     : Series  (diagnosed_diabetes)
        'multiclass' : Series  (diabetes_stage)
        'regression' : Series  (diabetes_risk_score)
    }
    """
    targets = {}
    required = {
        "binary"     : "diagnosed_diabetes",
        "multiclass" : "diabetes_stage",
        "regression" : "diabetes_risk_score",
    }
    for task, col in required.items():
        if col in df.columns:
            targets[task] = df[col]
            print(f"✅ Target for '{task}': '{col}'")
        else:
            print(f"⚠️  Column '{col}' not found – skipping '{task}'.")
    return targets


def get_feature_matrix(df: pd.DataFrame,
                        target_cols: list[str]) -> pd.DataFrame:
    """Return the feature matrix by dropping the target columns."""
    return df.drop(columns=[c for c in target_cols if c in df.columns])


# ──────────────────────────────────────────────
# 8. SAVE PROCESSED DATA
# ──────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save a processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved processed data → {path}")
