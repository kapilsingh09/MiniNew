"""
╔══════════════════════════════════════════════════════════════╗
║            AI DATA ANALYSIS — pre_process.py                 ║
║                                                              ║
║  Receives a DataFrame, target column, and task type from     ║
║  main.py.  Uses Gemini to produce a structured analysis      ║
║  report and saves it to the project folder.                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from google import genai


# ─────────────────────────────────────────────
# CORRELATION-BASED DROP DETECTION
# ─────────────────────────────────────────────
def get_highly_correlated_columns(df: pd.DataFrame, threshold: float = 0.90):
    """Return (columns_to_drop, pair_details) for correlations above *threshold*."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return [], []

    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    pairs = []
    for col in upper_tri.columns:
        correlated_with = upper_tri.index[upper_tri[col] > threshold].tolist()
        for partner in correlated_with:
            to_drop.append(col)
            pairs.append({
                "drop": col,
                "keep": partner,
                "correlation": round(corr_matrix.loc[col, partner], 4),
            })
    return list(set(to_drop)), pairs


# ─────────────────────────────────────────────
# FULL DATAFRAME ANALYSIS
# ─────────────────────────────────────────────
def analyze_dataframe(df: pd.DataFrame, corr_drop_cols: list, corr_pairs: list) -> dict:
    """Return a dictionary with an exhaustive statistical profile of *df*."""
    analysis = {}

    # ── Basic info
    analysis["head"] = df.head().to_string()
    analysis["tail"] = df.tail().to_string()
    analysis["shape"] = df.shape
    analysis["rows"] = len(df)
    analysis["columns_count"] = len(df.columns)
    analysis["columns"] = df.columns.tolist()
    analysis["dtypes"] = df.dtypes.to_string()
    analysis["dtype_count"] = df.dtypes.value_counts().to_string()
    analysis["size"] = df.size
    analysis["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 4)

    # ── Column type groups
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    boolean_cols = df.select_dtypes(include="bool").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    analysis["numeric_columns"] = numeric_cols
    analysis["categorical_columns"] = categorical_cols
    analysis["boolean_columns"] = boolean_cols
    analysis["datetime_columns"] = datetime_cols

    # ── Global stats
    analysis["describe"] = df.describe(include="all").to_string()
    analysis["skewness"] = df.skew(numeric_only=True).to_string()
    analysis["kurtosis"] = df.kurt(numeric_only=True).to_string()
    analysis["correlation"] = df.corr(numeric_only=True).to_string()
    analysis["covariance"] = df.cov(numeric_only=True).to_string()

    # ── Null summary
    total_nulls = int(df.isnull().sum().sum())
    analysis["total_null_values"] = total_nulls
    if total_nulls > 0:
        analysis["null_values"] = df.isnull().sum().to_string()
        analysis["null_percentage"] = (df.isnull().sum() / len(df) * 100).to_string()

    # ── Duplicate summary
    total_dupes = int(df.duplicated().sum())
    analysis["total_duplicates"] = total_dupes
    analysis["has_duplicates"] = total_dupes > 0
    if total_dupes > 0:
        analysis["duplicate_rows_preview"] = df[df.duplicated()].head(5).to_string()

    # ── Special flags
    analysis["constant_columns"] = [c for c in df.columns if df[c].nunique() == 1]
    analysis["high_cardinality_columns"] = [c for c in df.columns if df[c].nunique() > 50]
    analysis["low_cardinality_columns"] = [c for c in df.columns if df[c].nunique() < 10]
    analysis["zeros"] = (df == 0).sum().to_string()
    analysis["negative_values"] = (df.select_dtypes(include="number") < 0).sum().to_string()
    analysis["infinite_values"] = np.isinf(df.select_dtypes(include="number")).sum().to_string()

    # ── Pre-computed correlation drops
    analysis["highly_correlated_drop_candidates"] = corr_drop_cols
    analysis["correlation_pairs"] = corr_pairs

    # ── Per-column deep analysis
    per_column = {}
    for col in df.columns:
        col_data = df[col]
        info = {}
        info["dtype"] = str(col_data.dtype)

        # Null analysis
        null_count = int(col_data.isnull().sum())
        info["null_count"] = null_count
        if null_count > 0:
            info["null_percentage"] = round(null_count / len(df) * 100, 2)
            info["null_action_needed"] = True
            if col in numeric_cols:
                skew_val = col_data.skew()
                if abs(skew_val) > 1:
                    info["null_suggestion"] = f"MedianImputer (skew={round(skew_val, 2)}, skewed)"
                else:
                    info["null_suggestion"] = f"MeanImputer (skew={round(skew_val, 2)}, near-normal)"
            elif col in categorical_cols:
                info["null_suggestion"] = "ModeImputer (categorical column)"
            elif col in datetime_cols:
                info["null_suggestion"] = "ForwardFill or flag as missing datetime"
            else:
                info["null_suggestion"] = "Inspect manually"
        else:
            info["null_action_needed"] = False

        # Duplicate values
        dup_count = int(col_data.duplicated().sum())
        info["duplicate_value_count"] = dup_count
        if dup_count > 0:
            info["duplicate_percentage"] = round(dup_count / len(df) * 100, 2)

        # Unique values
        info["unique_count"] = int(col_data.nunique())
        info["unique_percentage"] = round(col_data.nunique() / len(df) * 100, 2)

        # Cardinality tag
        if col_data.nunique() == 1:
            info["cardinality_tag"] = "CONSTANT — drop this column"
        elif col_data.nunique() == 2:
            info["cardinality_tag"] = "BINARY"
        elif col_data.nunique() <= 5:
            info["cardinality_tag"] = "LOW (<=5) — OneHotEncoder"
        elif col_data.nunique() <= 15:
            info["cardinality_tag"] = "MEDIUM (6-15) — OrdinalEncoder or TargetEncoder"
        else:
            info["cardinality_tag"] = "HIGH (>15) — TargetEncoder or HashingEncoder"

        # Numeric-only stats
        if col in numeric_cols:
            info["mean"] = round(col_data.mean(), 4)
            info["median"] = round(col_data.median(), 4)
            info["std"] = round(col_data.std(), 4)
            info["min"] = round(col_data.min(), 4)
            info["max"] = round(col_data.max(), 4)
            info["skew"] = round(col_data.skew(), 4)
            info["kurt"] = round(col_data.kurt(), 4)

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = int(((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum())
            info["outlier_count"] = outlier_count
            if outlier_count > 0:
                info["outlier_percentage"] = round(outlier_count / len(df) * 100, 2)
                info["outlier_action"] = "RobustScaler or Winsorize (IQR clip)"

            skew = abs(col_data.skew())
            if skew > 1 and outlier_count > 0:
                info["scaling_suggestion"] = "Log1p transform + RobustScaler"
            elif skew > 1:
                info["scaling_suggestion"] = "Log1p then StandardScaler"
            elif outlier_count > 0:
                info["scaling_suggestion"] = "RobustScaler"
            else:
                info["scaling_suggestion"] = "StandardScaler"

            zero_count = int((col_data == 0).sum())
            neg_count = int((col_data < 0).sum())
            inf_count = int(np.isinf(col_data).sum())
            if zero_count > 0:
                info["zero_count"] = zero_count
            if neg_count > 0:
                info["negative_count"] = neg_count
                info["negative_warning"] = "Log transform will fail — use Log1p or shift first"
            if inf_count > 0:
                info["infinite_count"] = inf_count
                info["infinite_action"] = "Replace inf with NaN then impute"

        # Categorical-only stats
        if col in categorical_cols:
            info["top_5_values"] = col_data.value_counts().head(5).to_dict()
            info["mode"] = str(col_data.mode()[0]) if not col_data.mode().empty else "N/A"
            if null_count / len(df) > 0.4:
                info["drop_suggestion"] = (
                    f"DROP — {round(null_count / len(df) * 100, 1)}% nulls in categorical column"
                )

        per_column[col] = info

    analysis["per_column_analysis"] = per_column
    return analysis


# ─────────────────────────────────────────────
# GEMINI PROMPT
# ─────────────────────────────────────────────
def build_prompt(target_column: str, task_type: str, corr_pairs: list) -> str:
    """Return the full prompt template for Gemini analysis."""
    return f"""You are an elite Senior Data Scientist and ML Engineer.

You have been given a full DataFrame analysis. Your job is to produce a STRUCTURED, ACTIONABLE preprocessing and training pipeline specification.

This output will be consumed directly by another AI agent that will write the preprocessing + training code.
So every instruction MUST be:
- Specific (exact column names)
- Justified (why this action)
- Ordered (steps in the right sequence)

──────────────────────────────────────────────
TASK TYPE (AUTO-DETECTED): {task_type.upper()}
TARGET COLUMN: {target_column}
──────────────────────────────────────────────

HIGHLY CORRELATED COLUMNS TO DROP (pre-computed, threshold=0.90):
{corr_pairs}
These were detected automatically. Confirm or override based on domain reasoning.

──────────────────────────────────────────────
ANALYSIS SECTIONS TO COVER:
──────────────────────────────────────────────

═══════════════════════════════════════════════
            DATASET OVERVIEW
═══════════════════════════════════════════════
- Rows, columns, memory usage
- General data quality score (0-100)
- Is this dataset ML-ready as-is? Yes/No + reason

═══════════════════════════════════════════════
          DATA QUALITY ISSUES
═══════════════════════════════════════════════
- List all issues found: nulls, duplicates, constants, infinities, negatives
- Severity: Critical / Warning / Info for each

═══════════════════════════════════════════════
          MISSING VALUE ANALYSIS
═══════════════════════════════════════════════
For each column with nulls:
  Column | % Missing | Recommended Strategy | Technique

═══════════════════════════════════════════════
          DUPLICATE ANALYSIS
═══════════════════════════════════════════════
- Count and % of duplicates
- Action: Drop or Keep, with justification

═══════════════════════════════════════════════
        NUMERICAL FEATURE ANALYSIS
═══════════════════════════════════════════════
For each numeric column:
  - Range, mean, skewness, outliers present?
  - Recommended scaling: StandardScaler / MinMaxScaler / RobustScaler / Log Transform
  - Justification for each choice

═══════════════════════════════════════════════
       CATEGORICAL FEATURE ANALYSIS
═══════════════════════════════════════════════
For each categorical column:
  - Cardinality (unique count)
  - Recommended encoding:
      * Low cardinality (<=5): OneHotEncoder
      * Medium cardinality (6-15): OrdinalEncoder or TargetEncoder
      * High cardinality (>15): TargetEncoder or HashingEncoder
      * Binary: LabelEncoder
  - Is this an INPUT feature or the OUTPUT (target)?
  - For the TARGET column in classification: LabelEncoder if not already numeric

═══════════════════════════════════════════════
           OUTLIER ANALYSIS
═══════════════════════════════════════════════
For each numeric column:
  - Outliers detected? (IQR method)
  - Action: Clip / Winsorize / Log Transform / Keep
  - Justification

═══════════════════════════════════════════════
     CORRELATION & MULTICOLLINEARITY
═══════════════════════════════════════════════
- Confirm columns to drop from the pre-computed list above
- Any additional drops based on domain reasoning
- VIF risk columns (if any)

═══════════════════════════════════════════════
       FEATURE ENGINEERING IDEAS
═══════════════════════════════════════════════
- Suggest new features to create from existing ones
- Date/time decomposition if applicable
- Interaction terms or ratios worth trying
- Binning suggestions for skewed columns

═══════════════════════════════════════════════
      PREPROCESSING PIPELINE (ORDERED)
═══════════════════════════════════════════════
Give exact step-by-step pipeline in this format:

Step 1 | Action | Columns | Technique | Reason
Step 2 | ...

═══════════════════════════════════════════════
          COLUMNS TO DROP
═══════════════════════════════════════════════
Final consolidated drop list with reason for each:
  Column | Reason

═══════════════════════════════════════════════
         ML READINESS SCORE
═══════════════════════════════════════════════
Score: X/100
Reasoning: (brief)
After preprocessing estimated score: Y/100

═══════════════════════════════════════════════
        RECOMMENDED ML MODELS
═══════════════════════════════════════════════
Task: {task_type.upper()}

Top 3 models with rationale:
1. Model | Why it suits this data
2. Model | Why it suits this data
3. Model | Why it suits this data

Baseline model to start with: (e.g., LogisticRegression / LinearRegression)
Hyperparameter tuning priority: High / Medium / Low

═══════════════════════════════════════════════
         RISKS & LIMITATIONS
═══════════════════════════════════════════════
- Data leakage risks
- Class imbalance (for classification)
- Overfitting risk (small dataset?)
- Any ethical/bias risks in features

═══════════════════════════════════════════════
            FINAL SUMMARY
═══════════════════════════════════════════════
- 5-line summary of what this dataset is, what the task is, and the preprocessing plan
- One-line recommendation for the downstream AI agent

CRITICAL RULES:
✓ Use ONLY data provided — no hallucination
✓ Use EXACT column names from the dataset
✓ Every recommendation must have a WHY
✓ Downstream AI will implement this directly — be precise
✓ Task type is {task_type.upper()} — all model and encoding decisions must reflect this
"""


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT — called from main.py
# ─────────────────────────────────────────────
def run_analysis(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    project_dir: str,
) -> str:
    """
    Run the full AI analysis pipeline and return the path to the saved report.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment. Check your .env file.")

    # 1. Correlation analysis
    corr_drop_cols, corr_pairs = get_highly_correlated_columns(df, threshold=0.90)
    print(f"  🔗 Highly correlated drop candidates: {corr_drop_cols or 'None'}")

    # 2. Full dataframe analysis
    print("  📊 Running statistical analysis...")
    analysis_result = analyze_dataframe(df, corr_drop_cols, corr_pairs)

    # 3. Build prompt
    prompt = build_prompt(target_column, task_type, corr_pairs)
    analysis_text = "\n".join([
        f"[{key.upper()}]\n{value}\n"
        for key, value in analysis_result.items()
    ])
    full_prompt = (
        f"{prompt}\n\nHere is the full dataset analysis:\n{analysis_text}\n\n"
        f"Target Column: '{target_column}'\nTask Type: {task_type.upper()}"
    )

    # 4. Call Gemini
    client = genai.Client(api_key=api_key)
    print("  🤖 Calling Gemini for AI analysis...")
    response_text = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
            )
            response_text = response.text
            break
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("  Retrying in 3 seconds...")
                time.sleep(3)
            else:
                raise

    # 5. Save report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_header = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   AUTOMATED DATA ANALYSIS REPORT                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Generated   : {timestamp}
Target Col  : {target_column}
Task Type   : {task_type.upper()}
Dataset     : {df.shape[0]} rows × {df.shape[1]} columns
Memory      : {round(df.memory_usage(deep=True).sum() / 1024**2, 2)} MB
Corr Drops  : {corr_drop_cols if corr_drop_cols else "None detected"}

{'─' * 80}

"""

    formatted_report = (
        report_header
        + response_text
        + f"\n\n{'─' * 80}\nReport End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    report_path = os.path.join(project_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(formatted_report)

    print(f"  ✅ Analysis complete  |  Task: {task_type.upper()}  |  Target: {target_column}")
    return report_path