"""Shared cleaning code for the car price model.

This lives in its own file so the saved model can be loaded from any
program (the saved pipeline needs to import `clean_data` by name).
"""
import numpy as np
import pandas as pd

# Columns stored as text in the CSV because of '?' but really numeric
NUMERIC_COLS = [
    'symboling', 'normalized-losses', 'wheel-base', 'length', 'width',
    'height', 'curb-weight', 'engine-size', 'bore', 'stroke',
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
    'highway-mpg',
]

# Word columns that have a natural numeric meaning
WORD_TO_NUMBER = {
    'num-of-doors': {'two': 2, 'four': 4},
    'num-of-cylinders': {
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'eight': 8, 'twelve': 12,
    },
}

# Text columns that will be one-hot encoded
CATEGORICAL_COLS = [
    'make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
    'engine-location', 'engine-type', 'fuel-system',
]

# Everything the model treats as a number after cleaning
NUMERIC_FEATURES = NUMERIC_COLS + list(WORD_TO_NUMBER)


def clean_data(df):
    """Turn raw rows into model-ready types.

    This only fixes data types. It learns nothing from the data,
    so it is safe to run on any split without leakage.
    """
    df = df.copy()

    # '?' means missing
    df = df.replace('?', np.nan)

    # Words -> numbers (e.g. 'four' -> 4); safe to run twice
    for col, mapping in WORD_TO_NUMBER.items():
        if col in df.columns:
            df[col] = df[col].map(
                lambda v, m=mapping: m.get(str(v).strip().lower(), v)
            )

    # Text numbers -> real numbers; anything unreadable becomes NaN
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
