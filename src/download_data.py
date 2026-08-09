"""Download the public IBM Telco Customer Churn dataset."""

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = (
    "https://community.watsonanalytics.com/wp-content/uploads/2015/03/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
OUTPUT_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def download_dataset() -> Path:
    """Download the dataset if it is not already present."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        print(f"Dataset already exists: {OUTPUT_PATH}")
        return OUTPUT_PATH

    print("Downloading IBM Telco Customer Churn dataset...")
    urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"Saved dataset to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_dataset()
