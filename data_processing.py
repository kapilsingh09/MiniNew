from google import genai
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from main import load_data, what_to_predict
import os, json, re, time, threading, itertools, sys
import pandas as pd

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
load_dotenv()
df            = load_data()
target_column = what_to_predict()
client        = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ─────────────────────────────────────────────
# SPINNER
# ─────────────────────────────────────────────
def spinner_while(label="Thinking"):
    stop_event = threading.Event()
    def spin():
        for frame in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
            if stop_event.is_set(): break
            sys.stdout.write(f"\r{frame}  {label}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write(f"\r✅  {label} done!          \n")
        sys.stdout.flush()
    threading.Thread(target=spin).start()
    return stop_event

# ─────────────────────────────────────────────
# GEMINI CALL (with retry)
# ─────────────────────────────────────────────
def call_gemini(prompt, label="Gemini"):
    stop = spinner_while(label)
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            stop.set()
            return r.text.strip()
        except Exception as e:
            stop.set()
            print(f"\n⚠️  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10)
                stop = spinner_while(f"Retrying {label}")
            else:
                raise

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────
def drop_columns(df, cols):
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(f"  🗑️  Dropping: {cols}")
    return df.drop(columns=cols)

def apply_imputer(df, col, strategy):
    if col not in df.columns or col == target_column:
        return df
    df[col] = SimpleImputer(strategy=strategy).fit_transform(df[[col]])
    print(f"  🔧 Imputed  [{col}] → {strategy}")
    return df

def apply_encoder(df, col, method):
    if col not in df.columns or col == target_column:
        return df
    if method == "onehot":
        enc     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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
    print(f"  🔤 Encoded  [{col}] → {method}")
    return df

def apply_scaler(df, cols, method):
    cols = [c for c in cols if c in df.columns and c != target_column]
    if not cols: return df
    scaler = {"standard": StandardScaler(),
               "minmax":   MinMaxScaler(),
               "robust":   RobustScaler()}[method]
    df[cols] = scaler.fit_transform(df[cols])
    print(f"  📏 Scaled   {cols} → {method}")
    return df

# ─────────────────────────────────────────────
# LOAD ANALYSIS REPORT
# ─────────────────────────────────────────────
file_path = r"C:\Users\karan\Desktop\Sleepyy\analysis_report.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    analysis_report = f.read()[:15000]

# ─────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# GEMINI 1 — PREPROCESSING PLAN
# ─────────────────────────────────────────────
prompt_1 = f"""
You are a Senior ML Engineer.

Return a STRICT JSON preprocessing plan based on the dataset analysis report below.

RULES:
- Return ONLY valid raw JSON — no markdown, no ```json, no explanation
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
Train shape     : {X_train.shape}
Test shape      : {X_test.shape}

Dataset head:
{df.head().to_string()}

ANALYSIS REPORT:
{analysis_report}
"""

raw_1 = call_gemini(prompt_1, "Gemini 1 — building preprocessing plan")
raw_1 = re.sub(r"```json|```", "", raw_1).strip()
plan  = json.loads(raw_1)

print("\n📋 PREPROCESSING PLAN:")
print(json.dumps(plan, indent=2))

# ─────────────────────────────────────────────
# EXECUTE PREPROCESSING
# ─────────────────────────────────────────────
print("\n⚙️  Applying preprocessing...\n")
clean_df = df.copy()

clean_df = drop_columns(clean_df, plan.get("drop_columns", []))

for item in plan.get("impute", []):
    clean_df = apply_imputer(clean_df, item["column"], item["strategy"])

for item in plan.get("encode", []):
    clean_df = apply_encoder(clean_df, item["column"], item["method"])

for method, cols in plan.get("scale", {}).items():
    clean_df = apply_scaler(clean_df, cols, method)

# encode target for classification if still object
if plan.get("task_type") == "classification":
    if clean_df[target_column].dtype == object:
        clean_df[target_column] = LabelEncoder().fit_transform(
            clean_df[target_column].astype(str)
        )
        print(f"  🎯 Target  [{target_column}] → LabelEncoder")

print(f"\n✅ Clean dataset shape: {clean_df.shape}")
print(f"   Nulls remaining    : {int(clean_df.isnull().sum().sum())}")

# ─────────────────────────────────────────────
# GEMINI 2 — PIPELINE REVIEW
# ─────────────────────────────────────────────
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

Return ONLY raw JSON — no markdown, no explanation:
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

raw_2  = call_gemini(prompt_2, "Gemini 2 — reviewing pipeline")
raw_2  = re.sub(r"```json|```", "", raw_2).strip()
review = json.loads(raw_2)

print("\n📊 PIPELINE REVIEW:")
print(json.dumps(review, indent=2))

# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
clean_df.to_csv("clean_dataset.csv", index=False)

final_models = [m["model"] for m in review.get("final_models", [])]

with open("pipeline_plan.json", "w") as f:
    json.dump({
        "preprocessing_plan": plan,
        "pipeline_review":    review,
        "final_models":       final_models,
        "clean_shape":        list(clean_df.shape),
        "target_column":      target_column,
        "task_type":          plan.get("task_type")
    }, f, indent=2)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  💾 clean_dataset.csv   saved")
print(f"  📁 pipeline_plan.json  saved")
print(f"  ✅ Pipeline OK : {review.get('pipeline_ok')}")
print(f"  ⚠️  Issues     : {review.get('issues', [])}")
print(f"  🎯 Verdict     : {review.get('verdict', '')}")
print(f"  🤖 Train these : {final_models}")
print(f"{'─'*50}")