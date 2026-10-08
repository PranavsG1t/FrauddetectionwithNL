from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "rawdata" 

DTYPES = {
    "cc_num": "int64",
    "merchant": "category",
    "category": "category",
    "amt": "float32",
    "gender": "category",
    "city":"category",
    "state": "category",
    "zip": "int32",
    "lat": "float64",
    "long": "float64",
    "city_pop":"int64",
    "job": "category",
    "unix_time":"int64",
    "merch_lat": "float64",
    "merch_long": "float64",
    "is_fraud": "int8"
}

def load_split(name: str) -> pd.DataFrame:
    """Load 'train' or 'test'split"""
    filename = "fraudTrain.csv" if name == "train" else "fraudTest.csv"
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Please place the file in the rawdata/ directory.")
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["trans_date_trans_time","dob"],
        dtype={k: v for k, v in DTYPES.items()},
    )
    return df

def load_both() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split("train"), load_split("test")

if __name__ == "__main__":
    train = load_split("train")
    print(train.shape)
    print(train["is_fraud"].value_counts(normalize=True))
    print(train.dtypes)