"""
+==============================================================+
|          AI PREPROCESSING - data_processing.py               |
|                                                              |
|  Receives raw DataFrame + analysis report from main.py.      |
|  Uses two Gemini calls:                                      |
|    AI-1  ->  Build a preprocessing plan (JSON)                |
|    AI-2  ->  Review the applied pipeline                      |
|  Returns the clean DataFrame, plan, and review.              |
+==============================================================+
"""

import os
import sys
import json
import re
import time
import threading
import itertools
import pandas as pd
from google import genai
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
)


# _____________________________________________
# SPINNER (console progress indicator)
# _____________________________________________
def spinner_while(label: str = "Thinking"):
    """Start a console spinner. Returns a threading.Event to stop it."""
    stop_event = threading.Event()

    def spin():
        for frame in itertools.cycle(["|", "/", "-", "\\"]):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r{frame}  {label}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write(f"\r[OK] {label} done!          \n")
        sys.stdout.flush()

    threading.Thread(target=spin, daemon=True).start()
    return stop_event


# _____________________________________________
# GEMINI CALL (with retry + spinner)
# _____________________________________________
def call_gemini(client, prompt: str, label: str = "Gemini", model_name: str = "gemini-3-flash-preview") -> str:
    """Call Gemini with retries and a spinner."""
    stop = spinner_while(label)
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            stop.set()
            return r.text.strip()
        except Exception as e:
            stop.set()
            print(f"\n[WARN] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(10)
                stop = spinner_while(f"Retrying {label}")
            else:
                raise


# _____________________________________________
# PREPROCESSING FUNCTIONS
# _____________________________________________
def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(f"  [DROP] Dropping: {cols}")
    return df.drop(columns=cols)


def apply_imputer(df: pd.DataFrame, col: str, strategy: str, target: str) -> pd.DataFrame:
    if col not in df.columns or col == target:
        return df
    df[col] = SimpleImputer(strategy=strategy).fit_transform(df[[col]])
    print(f"  [IMPUTE] Imputed  [{col}] -> {strategy}")
    return df


def apply_encoder(df: pd.DataFrame, col: str, method: str, target: str) -> pd.DataFrame:
    if col not in df.columns or col == target:
        return df
    if method == "onehot":
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(df[[col]])
        new_cols = [f"{col}_{c}" for c in enc.categories_[0]]
        df = df.drop(columns=[col])
        df[new_cols] = encoded
    elif method == "ordinal":
        df[col] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(df[[col]])
    elif method == "label":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    print(f"  [ENCODE] Encoded  [{col}] -> {method}")
    return df


def apply_scaler(df: pd.DataFrame, cols: list, method: str, target: str) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns and c != target]
    if not cols:
        return df
    scaler = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }[method]
    df[cols] = scaler.fit_transform(df[cols])
    print(f"  [SCALE] Scaled   {cols} -> {method}")
    return df


# _____________________________________________
# PUBLIC ENTRY POINT - called from main.py
# _____________________________________________
def run_preprocessing(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    report_path: str,
    project_dir: str,
    model_name: str = "gemini-3-flash-preview",
) -> tuple:
    """
    Run the two-AI preprocessing pipeline.
    Returns (clean_df, plan_dict, review_dict).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found. Check your .env file.")
    client = genai.Client(api_key=api_key)

    # __ Load the analysis report
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        analysis_report = f.read()[:15000]

    # ==========================================
    # AI-1 - PREPROCESSING PLAN
    # ==========================================
    prompt_1 = f"""
You are a Senior ML Engineer.

Return a STRICT JSON preprocessing plan based on the dataset analysis report below.

RULES:
- Return ONLY valid raw JSON - no markdown, no ```json, no explanation
- Use EXACT column names from the dataset
- Never include the target column in drop/impute/encode/scale lists

JSON FORMAT:
{{
  "task_type": "classification" or "regression",
  "target_column": "{target_column}",
  "drop_columns": [],
  "impute": [
    {{"column": "col", "strategy": "mean" | "median" | "most_frequent"}}
  ],
  "encode": [
    {{"column": "col", "method": "onehot" | "ordinal" | "label"}}
  ],
  "scale": {{
    "standard": [],
    "minmax": [],
    "robust": []
  }},
  "models": ["Model1", "Model2", "Model3"]
}}

Dataset columns : {df.columns.tolist()}
Target column   : {target_column}
Dataset shape   : {df.shape}

Dataset head:
{df.head().to_string()}

ANALYSIS REPORT:
{analysis_report}
"""

    raw_1 = call_gemini(client, prompt_1, "AI-1 - Building preprocessing plan", model_name)
    raw_1 = re.sub(r"```json|```", "", raw_1).strip()
    plan = json.loads(raw_1)

    print("\n[PLAN] PREPROCESSING PLAN:")
    print(json.dumps(plan, indent=2))

    # ==========================================
    # EXECUTE PREPROCESSING
    # ==========================================
    print("\n[RUN] Applying preprocessing...\n")
    clean_df = df.copy()

    clean_df = drop_columns(clean_df, plan.get("drop_columns", []))

    for item in plan.get("impute", []):
        clean_df = apply_imputer(clean_df, item["column"], item["strategy"], target_column)

    for item in plan.get("encode", []):
        clean_df = apply_encoder(clean_df, item["column"], item["method"], target_column)

    for method, cols in plan.get("scale", {}).items():
        clean_df = apply_scaler(clean_df, cols, method, target_column)

    # Encode target for classification if still object
    if plan.get("task_type") == "classification":
        if clean_df[target_column].dtype == object:
            clean_df[target_column] = LabelEncoder().fit_transform(
                clean_df[target_column].astype(str)
            )
            print(f"  [TARGET] Target  [{target_column}] -> LabelEncoder")

    print(f"\n[OK] Clean dataset shape: {clean_df.shape}")
    print(f"     Nulls remaining  : {int(clean_df.isnull().sum().sum())}")

    # ==========================================
    # AI-2 - PIPELINE REVIEW
    # ==========================================
    prompt_2 = f"""
You are a Senior ML Engineer doing a final pipeline review.

The following preprocessing was applied. Check if everything is correct.

ORIGINAL PLAN:
{json.dumps(plan, indent=2)}

CLEAN DATASET STATE:
Shape            : {clean_df.shape}
Columns          : {clean_df.columns.tolist()}
Nulls remaining  : {int(clean_df.isnull().sum().sum())}
Dtypes:
{clean_df.dtypes.to_string()}

Stats:
{clean_df.describe(include='all').to_string()[:4000]}

TASK: {plan.get("task_type", "classification").upper()}
TARGET: {target_column}

Return ONLY raw JSON - no markdown, no explanation:
{{
  "pipeline_ok": true or false,
  "issues": [],
  "nulls_remaining": {{}},
  "unencoded_columns": [],
  "final_models": [
    {{"model": "ModelName", "reason": "one line"}},
    {{"model": "ModelName", "reason": "one line"}},
    {{"model": "ModelName", "reason": "one line"}}
  ],
  "verdict": "one line final verdict"
}}
"""

    raw_2 = call_gemini(client, prompt_2, "AI-2 - Reviewing pipeline", model_name)
    raw_2 = re.sub(r"```json|```", "", raw_2).strip()
    review = json.loads(raw_2)

    print("\n[REVIEW] PIPELINE REVIEW:")
    print(json.dumps(review, indent=2))

    # ==========================================
    # SAVE OUTPUTS
    # ==========================================
    clean_csv_path = os.path.join(project_dir, "clean_dataset.csv")
    clean_df.to_csv(clean_csv_path, index=False)

    final_models = [m["model"] for m in review.get("final_models", [])]

    pipeline_json_path = os.path.join(project_dir, "pipeline_plan.json")
    with open(pipeline_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "preprocessing_plan": plan,
            "pipeline_review": review,
            "final_models": final_models,
            "clean_shape": list(clean_df.shape),
            "target_column": target_column,
            "task_type": plan.get("task_type"),
        }, f, indent=2)

    print(f"\n{'_' * 50}")
    print(f"  [SAVED] clean_dataset.csv")
    print(f"  [SAVED] pipeline_plan.json")
    print(f"  [OK]    Pipeline OK : {review.get('pipeline_ok')}")
    print(f"  [WARN]  Issues      : {review.get('issues', [])}")
    print(f"  [INFO]  Verdict     : {review.get('verdict', '')}")
    print(f"  [INFO]  Train these : {final_models}")
    print(f"{'_' * 50}")

    return clean_df, plan, review