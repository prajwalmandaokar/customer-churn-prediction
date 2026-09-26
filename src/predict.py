"""
predict.py
----------
Load the trained pipeline and score new customer records.

Usage as a script:
    python src/predict.py --input data/raw/customer_churn.csv --output reports/predictions.csv

Usage as a library:
    from src.predict import ChurnPredictor
    predictor = ChurnPredictor()
    predictor.predict_df(new_customers_df)
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data_preprocessing import clean_data, engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = ROOT / "models" / "model_columns.json"


class ChurnPredictor:
    def __init__(self, model_path=MODEL_PATH, columns_path=COLUMNS_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python src/train_model.py` first."
            )
        self.pipeline = joblib.load(model_path)
        with open(columns_path) as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        self.best_model_name = meta["best_model"]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a dataframe shaped like the raw dataset (same columns as
        data/raw/customer_churn.csv, minus Churn — Churn is ignored if present).
        Returns the input with two new columns: ChurnProbability, ChurnPrediction.
        """
        df = df.copy()
        has_target = "Churn" in df.columns
        if not has_target:
            df["Churn"] = "No"  # placeholder so clean_data() doesn't choke

        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)

        X = engineered[self.feature_columns]

        proba = self.pipeline.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # Note: clean_data() may drop duplicate rows, so the result is built
        # from the cleaned frame (not the original df) to keep lengths aligned.
        original_cols = [c for c in df.columns if c != "Churn"] if not has_target else list(df.columns)
        result = cleaned[original_cols].reset_index(drop=True)
        result["ChurnProbability"] = proba.round(4)
        result["ChurnPrediction"] = pd.Series(pred).map({1: "Yes", 0: "No"})
        return result

    def predict_single(self, customer: dict) -> dict:
        """Predict for one customer given as a dict of raw field values."""
        df = pd.DataFrame([customer])
        result = self.predict_df(df)
        return {
            "ChurnProbability": float(result.loc[0, "ChurnProbability"]),
            "ChurnPrediction": result.loc[0, "ChurnPrediction"],
        }


def main():
    parser = argparse.ArgumentParser(description="Score customers with the trained churn model.")
    parser.add_argument("--input", required=True, help="Path to a CSV shaped like the raw dataset.")
    parser.add_argument("--output", default=str(ROOT / "reports" / "predictions.csv"))
    args = parser.parse_args()

    predictor = ChurnPredictor()
    df = pd.read_csv(args.input)
    result = predictor.predict_df(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} customers using {predictor.best_model_name}.")
    print(f"Predicted churners: {(result['ChurnPrediction'] == 'Yes').sum():,}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
