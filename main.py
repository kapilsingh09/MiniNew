"""
+==============================================================+
|               AUTOMATED ML PIPELINE - main.py               |
|                                                              |
|  Single entry point that orchestrates:                       |
|    1. Data Loading                                           |
|    2. AI Analysis & Report  (pre_process.py)                 |
|    3. AI Preprocessing      (data_processing.py)             |
|    4. Model Training        (train_model.py)                 |
+==============================================================+
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv



# _____________________________________________
# CONSTANTS
# _____________________________________________
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(PROJECT_DIR, "csv_data")

# Change this to switch the Gemini model used across the entire pipeline
GEMINI_MODEL = "gemini-3-flash-preview"


# _____________________________________________
# DATA LOADING
# _____________________________________________
def load_data(csv_dir: str = CSV_DIR) -> pd.DataFrame:
    """Load the first CSV file found in *csv_dir*."""
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print("[ERROR] No CSV files found in:", csv_dir)
        sys.exit(1)

    if len(csv_files) > 1:
        print("\nMultiple CSV files found:")
        for i, f in enumerate(csv_files, 1):
            print(f"  [{i}] {f}")
        choice = input("Choose file number: ").strip()
        try:
            chosen = csv_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Using first file.")
            chosen = csv_files[0]
    else:
        chosen = csv_files[0]

    path = os.path.join(csv_dir, chosen)
    df = pd.read_csv(path)
    print(f"[OK] Loaded: {chosen}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return df


# _____________________________________________
# USER INPUT - TARGET COLUMN
# _____________________________________________
def ask_target_column(df: pd.DataFrame) -> str:
    """Prompt user to pick the target / prediction column."""
    print("\nColumns in your dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"  [{i}] {col}")
    choice = input("\nWhich column do you want to PREDICT? (name or number): ").strip()

    # accept by number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(df.columns):
            target = df.columns[idx]
        else:
            raise ValueError
    except ValueError:
        target = choice  # accept by name

    if target not in df.columns:
        print(f"[ERROR] '{target}' is not a valid column. Exiting.")
        sys.exit(1)

    print(f"[TARGET] Target column: {target}")
    return target


# _____________________________________________
# AUTO-DETECT TASK TYPE
# _____________________________________________
def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Automatically classify as 'classification' or 'regression'
    based on the target column's dtype and cardinality.
    """
    col = df[target]

    # Categorical / object / bool -> classification
    if col.dtype == "object" or col.dtype == "bool":
        return "classification"

    # Numeric with very few unique values -> classification
    nunique = col.nunique()
    if nunique <= 15:
        return "classification"

    # Otherwise -> regression
    return "regression"


# _____________________________________________
# MAIN PIPELINE
# _____________________________________________
def main():
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))

    print("=" * 60)
    print("       AUTOMATED ML PIPELINE")
    print("=" * 60)

    # __ Step 0 - Load data
    df = load_data()
    target = ask_target_column(df)
    task_type = detect_task_type(df, target)
    print(f"[DETECT] Auto-detected task type: {task_type.upper()}")

    confirm = input(f"\nProceed with task={task_type.upper()}, target='{target}'? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Aborted.")
        sys.exit(0)

    # __ Step 1 - AI Analysis & Report
    print("\n" + "=" * 60)
    print("  STEP 1 / 3 - AI Data Analysis")
    print("=" * 60)
    from pre_process import run_analysis
    report_path = run_analysis(df, target, task_type, PROJECT_DIR, GEMINI_MODEL)
    print(f"[SAVED] Report saved -> {report_path}")

    # __ Step 2 - AI Preprocessing & Clean Dataset
    print("\n" + "=" * 60)
    print("  STEP 2 / 3 - AI Preprocessing")
    print("=" * 60)
    from data_processing import run_preprocessing
    clean_df, plan, review = run_preprocessing(df, target, task_type, report_path, PROJECT_DIR, GEMINI_MODEL)
    print(f"[SAVED] Clean dataset saved -> {os.path.join(PROJECT_DIR, 'clean_dataset.csv')}")

    # __ Step 3 - Model Training
    print("\n" + "=" * 60)
    print("  STEP 3 / 3 - Model Training")
    print("=" * 60) 
    from train_model import run_training
    run_training(clean_df, target, task_type, review, PROJECT_DIR)

    # __ Done
    print("\n" + "=" * 60)
    print("  [DONE] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  All outputs saved in: {PROJECT_DIR}")
    print(f"     - analysis_report.txt   -- AI data analysis")
    print(f"     - clean_dataset.csv     -- Preprocessed data")
    print(f"     - pipeline_plan.json    -- Full pipeline spec")
    print(f"     - training_results.json -- Training metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()