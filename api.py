"""
+==============================================================+
|               FASTAPI SERVER - api.py                        |
|                                                              |
|  Simple FastAPI server to run the ML pipeline via HTTP       |
+==============================================================+
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pipeline imports
from pre_process import run_analysis
from data_processing import run_preprocessing
from train_model import run_training

# Create FastAPI app
app = FastAPI(
    title="Automated ML Pipeline API",
    description="Upload a CSV and get automated ML analysis, preprocessing, and model training",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = Path(PROJECT_DIR) / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
GEMINI_MODEL = "gemini-3-flash-preview"


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """Auto-detect if task is classification or regression"""
    col = df[target]
    if col.dtype == "object" or col.dtype == "bool":
        return "classification"
    nunique = col.nunique()
    if nunique <= 15:
        return "classification"
    return "regression"


def run_pipeline_auto(csv_path: str, target_column: str = None):
    """
    Run the complete ML pipeline on a CSV file.
    
    Args:
        csv_path: Path to CSV file
        target_column: Target column name (if None, uses last column)
    
    Returns:
        Dictionary with results
    """
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded: {os.path.basename(csv_path)}  ({df.shape[0]} rows x {df.shape[1]} cols)")
        
        # Determine target column
        if target_column is None:
            target_column = df.columns[-1]
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")
        
        # Detect task type
        task_type = detect_task_type(df, target_column)
        print(f"[DETECT] Auto-detected task type: {task_type.upper()}")
        
        # Step 1: AI Analysis
        print("\n[STEP 1/3] Running AI Data Analysis...")
        report_path = run_analysis(df, target_column, task_type, PROJECT_DIR, GEMINI_MODEL)
        
        # Step 2: AI Preprocessing
        print("[STEP 2/3] Running AI Preprocessing...")
        clean_df, plan, review = run_preprocessing(df, target_column, task_type, report_path, PROJECT_DIR, GEMINI_MODEL)
        
        # Step 3: Model Training
        print("[STEP 3/3] Training Models...")
        training_results = run_training(clean_df, target_column, task_type, review, PROJECT_DIR)
        
        # Load results from JSON files
        results = {
            "status": "success",
            "task_type": task_type,
            "target_column": target_column,
            "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        }
        
        # Load analysis report
        try:
            with open(os.path.join(PROJECT_DIR, "analysis_report.txt"), "r") as f:
                results["analysis_report"] = f.read()
        except:
            results["analysis_report"] = "Report not found"
        
        # Load pipeline plan
        try:
            with open(os.path.join(PROJECT_DIR, "pipeline_plan.json"), "r") as f:
                results["pipeline_plan"] = json.load(f)
        except:
            results["pipeline_plan"] = {}
        
        # Load training results
        try:
            with open(os.path.join(PROJECT_DIR, "training_results.json"), "r") as f:
                results["training_results"] = json.load(f)
        except:
            results["training_results"] = {}
        
        return results
    
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {str(e)}")
        raise


@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "🚀 Welcome to Automated ML Pipeline API",
        "description": "Upload a CSV to automatically analyze, preprocess, and train ML models",
        "endpoints": {
            "POST /predict": "Upload CSV and run pipeline",
            "POST /predict-path": "Run pipeline on file in csv_data folder",
            "GET /health": "Check API status",
            "GET /files": "List files in csv_data folder"
        },
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "✓ healthy",
        "service": "Automated ML Pipeline API",
        "version": "1.0.0"
    }


@app.get("/files")
def list_files():
    """List CSV files available in csv_data folder"""
    csv_dir = Path(PROJECT_DIR) / "csv_data"
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    return {
        "csv_data_folder": str(csv_dir),
        "files": csv_files,
        "count": len(csv_files)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), target_column: str = Query(None)):
    """
    Upload a CSV file and run the complete ML pipeline.
    
    Args:
        file: CSV file to upload
        target_column: Target column name (optional, defaults to last column)
    
    Returns:
        - status: success/error
        - task_type: classification/regression
        - target_column: the target column used
        - analysis_report: AI-generated data analysis
        - pipeline_plan: preprocessing strategy
        - training_results: model performance metrics
    """
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[INFO] Processing uploaded file: {file.filename}")
        
        # Run pipeline
        results = run_pipeline_auto(str(file_path), target_column)
        
        # Clean up temp file
        os.remove(file_path)
        
        return JSONResponse(
            status_code=200,
            content=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-path")
async def predict_from_path(file_name: str, target_column: str = Query(None)):
    """
    Run ML pipeline on a CSV file already in csv_data folder.
    
    Args:
        file_name: CSV file name in csv_data/ (e.g., "student_performance.csv")
        target_column: Target column name (optional, defaults to last column)
    """
    
    try:
        full_path = Path(PROJECT_DIR) / "csv_data" / file_name
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
        
        if not str(full_path).endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        print(f"[INFO] Processing file: {file_name}")
        
        # Run pipeline
        results = run_pipeline_auto(str(full_path), target_column)
        
        return JSONResponse(
            status_code=200,
            content=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server...")
    print("[INFO] Open browser at http://localhost:8000/docs for interactive API docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
