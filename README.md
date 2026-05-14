# 🚀 Automated ML Pipeline

An AI-powered end-to-end machine learning pipeline that **analyzes**, **preprocesses**, and **trains models** on any CSV dataset — automatically.

## ⚡ Quick Start (2 minutes)

### 1️⃣ Setup

```bash
# Install dependencies (one time only)
pip install -r requirements.txt

# Set your API key in .env
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 2️⃣ Run as CLI (Interactive)

```bash
# For interactive mode (asks you questions)
python main.py
```

### 3️⃣ Run as API Server (Recommended for production)

```bash
# Start the FastAPI server
python api.py

# Open browser: http://localhost:8000/docs
# Now you can upload CSVs and get results via HTTP!
```

## How It Works

```
python main.py
```

The pipeline runs 3 stages:

| Stage | Module | AI Calls | Output |
|-------|--------|----------|--------|
| 1. Analysis | `pre_process.py` | Gemini × 1 | `analysis_report.txt` |
| 2. Preprocessing | `data_processing.py` | Gemini × 2 | `clean_dataset.csv`, `pipeline_plan.json` |
| 3. Training | `train_model.py` | None | `training_results.json` |

## Setup (First Time Only)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Get your Google Gemini API key from https://makersuite.google.com/app/apikey

# 3. Add your API key to .env file (create if doesn't exist)
echo "GOOGLE_API_KEY=your_key_here" > .env

# 4. Put CSV files in csv_data/ folder
```

## Features

- **Auto task detection** — Classifies your target as classification or regression
- **AI-powered analysis** — Gemini generates a comprehensive statistical report
- **Smart preprocessing** — AI decides imputation, encoding, scaling strategies
- **Pipeline verification** — Second AI reviews the preprocessing for correctness
- **Multi-model training** — Trains 3+ models with cross-validation
- **Best model selection** — Automatically ranks and highlights the winner

## Project Structure

```
├── main.py              ← Entry point (run this)
├── pre_process.py       ← AI data analysis module
├── data_processing.py   ← AI preprocessing module
├── train_model.py       ← Model training module
├── csv_data/            ← Drop your CSV files here
├── .env                 ← API key config
└── outputs/
    ├── analysis_report.txt
    ├── clean_dataset.csv
    ├── pipeline_plan.json
    └── training_results.json
```

## 🌐 FastAPI Server

You can also run the ML pipeline as a REST API server instead of a CLI tool!

### Start the API Server

```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Start the server
python api.py
```

The server will start at `http://localhost:8000`

### Interactive API Documentation

Open your browser to **http://localhost:8000/docs** to see the Swagger UI with all available endpoints and try them out interactively!

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message & endpoint info |
| `/health` | GET | Check API status |
| `/files` | GET | List CSV files in csv_data/ |
| `/predict` | POST | Upload CSV and run pipeline |
| `/predict-path` | POST | Run pipeline on existing CSV file |

### Example Usage

**Upload a CSV file:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_data.csv" \
  -F "target_column=SalePrice"
```

**Use existing file from csv_data folder:**
```bash
curl -X POST "http://localhost:8000/predict-path?file_name=student_performance.csv&target_column=score"
```

**Response Example:**
```json
{
  "status": "success",
  "task_type": "regression",
  "target_column": "SalePrice",
  "data_shape": {"rows": 1460, "columns": 81},
  "analysis_report": "...",
  "pipeline_plan": {...},
  "training_results": {...}
}
```

### Query Parameters

- `target_column` (optional): Name of the column to predict. If not specified, uses the last column.

## 📂 Project Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point (interactive mode) |
| `api.py` | FastAPI server (REST API mode) |
| `pre_process.py` | AI data analysis module |
| `data_processing.py` | AI preprocessing module |
| `train_model.py` | Model training module |
| `csv_data/` | Drop your CSV files here |
| `.env` | Store your Google API key |
| `requirements.txt` | Python dependencies |

## 📊 What Gets Generated

After running the pipeline, you'll get 4 output files:

1. **`analysis_report.txt`** - Statistical analysis from Gemini AI
2. **`clean_dataset.csv`** - Preprocessed, ready-to-train data
3. **`pipeline_plan.json`** - Exact preprocessing steps used
4. **`training_results.json`** - Model performance metrics

## ✅ Next Steps

1. **First time setup:**
   - Copy your Google Gemini API key
   - Run: `pip install -r requirements.txt`
   - Create `.env` with your API key

2. **Choose your mode:**
   - **CLI mode**: `python main.py` → Interactive questions
   - **API mode**: `python api.py` → HTTP server at localhost:8000

3. **Add your data:**
   - Drop CSV files in the `csv_data/` folder
   - Or upload them via the API

4. **Run & get results:**
   - Results automatically saved to output files
   - View API docs at: http://localhost:8000/docs

## 📝 Requirements

- Python 3.8+
- Google Gemini API key (free at https://makersuite.google.com/app/apikey)
- Dependencies: pandas, numpy, scikit-learn, google-genai, python-dotenv, fastapi, uvicorn
