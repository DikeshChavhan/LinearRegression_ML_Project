import pandas as pd
import numpy as np

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the insurance model.
    Best-effort: ensures expected columns, encodes categorical cols, creates region dummies.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    expected = ["age", "bmi", "charges"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # sex mapping
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.lower().map({'male':0,'female':1}).fillna(0)

    # smoker mapping
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].astype(str).str.lower().map({'yes':1,'no':0}).fillna(0)

    # region dummies
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str).str.lower()
        regions = ['southeast','southwest','northeast','northwest']
        for r in regions:
            df['region_' + r] = (df['region'] == r).astype(int)
        df.drop(columns=['region'], inplace=True, errors='ignore')

    # numeric conversions
    for c in ['age','bmi','children']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # construct final columns: try to match expected order, append region dummies if present
    final_cols = list(["age", "bmi", "charges"])
    for r in ['southeast','southwest','northeast','northwest']:
        rn = 'region_' + r
        if rn in df.columns and rn not in final_cols:
            final_cols.append(rn)
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0
    return df[final_cols]
