from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "notebook" / "models" / "final_random_forest.joblib"

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
if not METADATA_PATH.exists():
    METADATA_PATH = PROJECT_ROOT / "notebook" / "models" / "model_metadata.json"


def prepare_features(df):
    df = df.copy()

    df["Date_Time"] = pd.to_datetime(
        df["Date_Time"],
        dayfirst=True,
        errors="raise"
    )

    df["Hour"] = df["Date_Time"].dt.hour
    df["Day_of_Week"] = df["Date_Time"].dt.day_of_week
    df["Month"] = df["Date_Time"].dt.month

    df["Time_Slot"] = (
        df["Date_Time"].dt.hour * 4
        + df["Date_Time"].dt.minute // 15
    )

    df["Is_Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

    return df


def predict(input_path, output_path):
    # Import the module so the custom transformer is available
    import src.preprocessing

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    df = pd.read_csv(input_path)
    df = prepare_features(df)

    features = metadata["features"]

    missing_features = [
        col for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required columns: {missing_features}"
        )

    predictions = model.predict(df[features])

    output = pd.DataFrame({
        "Date_Time": df["Date_Time"],
        "Predicted_Load_Type": predictions
    })

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Saved {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict industrial electricity load categories"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        default="results/new_predictions.csv",
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    predict(args.input, args.output)