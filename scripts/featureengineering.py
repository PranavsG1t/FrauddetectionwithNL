import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import matplotlib.pyplot as plt
from data.load_data import load_split
from data.preprocess import build_features, FEATURE_COLUMNS, DROP_COLUMNS

OUT_DIR = Path(__file__).resolve().parents[1] / "notebooks" 
OUT_DIR.mkdir(exist_ok=True)

def main():
    print("Loading data...")
    train = load_split("train")
    print(f"shape: {train.shape}")

    fraud_rate = train["is_fraud"].mean()
    print(f"Fraud rate: {fraud_rate:.4%}(expect well under 1%)")

    print("\n Building features on train...")
    train_featured, category_encoding = build_features(train)

    print("Loading test split and applying the same encoding (no leak)..")
    test = load_split("test")
    test_featured, _ = build_features(test, category_encoding=category_encoding)

    print(f"\nTrain Features shape: {train_featured.shape}")
    print(f"Test Features shape: {test_featured.shape}")

    print("\nFeature summary:(train):")
    print(train_featured[FEATURE_COLUMNS].describe())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    train_featured["is_fraud"].value_counts(normalize=True).plot(kind="bar", ax=axes[0, 0], title="Class balance")

    for label, group in train_featured.groupby("is_fraud"):
        axes[0, 1].hist(group["geo_distance_km"].clip(upper=500), bins=40, alpha=0.5, label=f"is_fraud={label}", density=True)
    axes[0, 1].set_title("Geo distance distribution")
    axes[0, 1].legend()

    for label, group in train_featured.groupby("is_fraud"):
        axes[1, 0].hist(group["mins_since_last_txn"].clip(upper=500), bins=40, alpha=0.5, label=f"is_fraud={label}", density=True)
    axes[1, 0].set_title("Minutes since last transaction by class")
    axes[1, 0].legend()

    for label, group in train_featured.groupby("is_fraud"):
        axes[1, 1].hist(group["amt_zscore"].clip(-3, 5), bins=40, alpha=0.5, label=f"is_fraud={label}", density=True)
    axes[1, 1].set_title("Amount z-score distribution by class")
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = OUT_DIR / "day1_eda.png"
    plt.savefig(fig_path,dpi=120)
    print(f"\nSaved EDA figure to {fig_path}")

    #featured data for 2nd modeling step
    OUT_DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "data"
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_featured.to_parquet(OUT_DATA_DIR / "train_featured.parquet")
    test_featured.to_parquet(OUT_DATA_DIR / "test_featured.parquet")
    print(f"\nSaved train_featured.parquet and test_featured.parquet to {OUT_DATA_DIR}")


if __name__ == "__main__":
    main()

