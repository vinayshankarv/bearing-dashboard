"""
utils.py — RCA_EfficientNet Dashboard Utilities  v2.0
======================================================
Live Prediction removed.
All metrics are loaded from disk — no fallback / synthetic / hardcoded data.
Any missing file raises an explicit, actionable error immediately.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Normal",
    "Inner Race Fault",
    "Outer Race Fault",
    "Ball Fault",
    "Cage Fault",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224

MODELS_DIR  = "models"
METRICS_DIR = "metrics"
ASSETS_DIR  = "assets"

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    """
    Raise ValueError if any required columns are missing from df.

    Parameters
    ----------
    df       : DataFrame to validate.
    required : Set of column name strings that must be present.
    filename : Name of the source file (used in the error message).
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{filename}' is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


# ─── Metrics Loaders ──────────────────────────────────────────────────────────

def load_model_comparison() -> pd.DataFrame:
    """
    Load the model comparison table from metrics/model_comparison.csv.

    Required columns : Model, Accuracy, F1 Score
    Optional columns : Precision, Recall  (used for radar chart / KPI tiles)

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        CSV does not exist at the expected path.
    ValueError
        One or more required columns are absent from the CSV.
    """
    path = os.path.join(METRICS_DIR, "model_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required metrics file not found: '{path}'\n"
            f"Generate model_comparison.csv during evaluation and place it "
            f"in the '{METRICS_DIR}/' directory."
        )
    df = pd.read_csv(path)
    _require_columns(df, {"Model", "Accuracy", "F1 Score"}, "model_comparison.csv")
    return df


def load_roc_data(model_name: str) -> pd.DataFrame:
    """
    Load ROC curve data from metrics/roc_<model_name>.csv.

    Required columns : FPR, TPR  (case-insensitive — normalised to UPPER on load)

    Parameters
    ----------
    model_name : One of CNN | ResNet50 | EfficientNet | DeiT | RCA_EfficientNet

    Returns
    -------
    pd.DataFrame  with columns FPR, TPR

    Raises
    ------
    FileNotFoundError
        CSV does not exist.
    ValueError
        Required columns are missing after normalisation.
    """
    filename = f"roc_{model_name}.csv"
    path     = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ROC data not found: '{path}'\n"
            f"Generate {filename} during evaluation and place it "
            f"in the '{METRICS_DIR}/' directory."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    _require_columns(df, {"FPR", "TPR"}, filename)
    return df


def load_pr_data(model_name: str) -> pd.DataFrame:
    """
    Load Precision-Recall curve data from metrics/pr_<model_name>.csv.

    Required columns : Recall, Precision  (capitalised on load)

    Parameters
    ----------
    model_name : One of CNN | ResNet50 | EfficientNet | DeiT | RCA_EfficientNet

    Returns
    -------
    pd.DataFrame  with columns Recall, Precision

    Raises
    ------
    FileNotFoundError
        CSV does not exist.
    ValueError
        Required columns are missing.
    """
    filename = f"pr_{model_name}.csv"
    path     = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PR data not found: '{path}'\n"
            f"Generate {filename} during evaluation and place it "
            f"in the '{METRICS_DIR}/' directory."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.capitalize()
    _require_columns(df, {"Recall", "Precision"}, filename)
    return df


def load_confusion_matrix(model_name: str) -> pd.DataFrame:
    """
    Load a confusion matrix from metrics/cm_<model_name>.csv.

    The CSV must be square (rows = true labels, columns = predicted labels)
    with class names as both the row index and column headers.

    Parameters
    ----------
    model_name : One of CNN | ResNet50 | EfficientNet | DeiT | RCA_EfficientNet

    Returns
    -------
    pd.DataFrame  (square, index = true labels, columns = predicted labels)

    Raises
    ------
    FileNotFoundError
        CSV does not exist.
    ValueError
        The matrix is not square.
    """
    filename = f"cm_{model_name}.csv"
    path     = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Confusion matrix not found: '{path}'\n"
            f"Generate {filename} during evaluation and place it "
            f"in the '{METRICS_DIR}/' directory."
        )
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"'{filename}' must be a square matrix. "
            f"Got shape: {df.shape}"
        )
    return df


def list_sample_images() -> list:
    """
    Return sorted paths of all image files found directly in ASSETS_DIR.

    Returns an empty list (not an error) when the directory is absent or empty.
    This function is retained for any future diagnostic/display use but is
    no longer called by the Live Prediction page (which has been removed).

    Returns
    -------
    list[str]  Sorted absolute or relative paths to image files.
    """
    if not os.path.isdir(ASSETS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [
        os.path.join(ASSETS_DIR, f)
        for f in sorted(os.listdir(ASSETS_DIR))
        if os.path.splitext(f)[1].lower() in exts
    ]
