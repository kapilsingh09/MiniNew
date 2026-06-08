# 🚀 Unified Automated ML Pipeline v2.0

An **AI-powered, parallel-processing** end-to-end machine learning pipeline that **analyzes**, **preprocesses**, and **trains optimized models** on any CSV dataset — **in a single unified route**.

## 🎯 Key Features

✅ **Single Unified Route** — All ML steps (EDA → preprocessing → training) in one `/predict` endpoint  
✅ **Parallel Processing** — EDA and feature engineering run concurrently for **fast results**  
✅ **AI-Powered** — Google Gemini API for intelligent preprocessing decisions  
✅ **Multi-Model Training** — 3+ models trained in parallel with cross-validation  
✅ **Auto Task Detection** — Automatically classifies as classification or regression  
✅ **Model Persistence** — Trained models saved and retrievable  
✅ **Production Ready** — FastAPI with CORS, error handling, and async support  

## ⚡ Quick Start (5 minutes)

### 1️⃣ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Get your Google Gemini API key from https://makersuite.google.com/app/apikey

# Create .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 2️⃣ Run as FastAPI Server

```bash
# Start the server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Open in browser: http://localhost:8000/docs
```

### 3️⃣ Upload CSV & Get Results

```bash
# Using curl
curl -X POST http://localhost:8000/predict \
  -F "file=@your_data.csv" \
  -F "target_column=target_name"
```

Or use the **interactive Swagger UI** at `http://localhost:8000/docs`

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED /predict ROUTE                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    STEP 0: Load CSV
         │
         ▼
    STEP 1: ⚡ Parallel EDA Analysis
         ├─► Task 1: Statistical Analysis
         └─► Task 2: Correlation Detection
         │
         ▼
    STEP 2: ⚡ AI Preprocessing (Parallel)
         ├─► Google Gemini: Generate plan
         └─► Apply preprocessing:
             ├─ Drop columns
             ├─ Handle missing values
             ├─ Encode categorical
             └─ Scale features
         │
         ▼
    STEP 3: ⚡ Parallel Model Training
         ├─► Train Model 1 (Classification/Regression)
         ├─► Train Model 2
         └─► Train Model 3+
         │
         ▼
    RETURN: Best Model + Metrics
```

## 📊 API Endpoints

### POST `/predict` — Unified ML Pipeline

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data.csv" \
  -F "target_column=price"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2024-05-31T10:30:00",
  "data_info": {
    "original_shape": [1000, 20],
    "final_shape": [1000, 18],
    "target_column": "price",
    "task_type": "regression"
  },
  "preprocessing": {
    "columns_dropped": ["id", "name"],
    "columns_encoded": ["category", "region"],
    "scaling_method": "standard"
  },
  "model_training": {
    "best_model": "RandomForestRegressor",
    "best_score": 0.8756,
    "train_test_split": {"train": [800, 15], "test": [200, 15]}
  },
  "model_file": "/path/to/trained_models/best_model_regression_20240531_103000.pkl"
}
```

### GET `/` — Health Check

```bash
curl http://localhost:8000/
```

### GET `/models` — List Trained Models

```bash
curl http://localhost:8000/models
```

## 🛠️ Project Structure

```
MiniNew/
├── api.py                  ← FastAPI server (NEW: Unified route)
├── pre_process.py          ← EDA analysis module
├── data_processing.py      ← AI preprocessing module
├── train_model.py          ← Model training with parallel support
├── main.py                 ← CLI entry point (legacy)
├── requirements.txt        ← Updated dependencies
├── README.md              ← This file
├── .env                    ← API key (create this)
├── temp_uploads/          ← Temp CSV storage
└── trained_models/        ← Saved model files
```

## 🚀 Parallelization Details

### EDA Analysis (Parallel)
- Task 1: Compute statistical metrics
- Task 2: Detect highly correlated columns
- **Speedup**: ~2x faster than sequential

### Model Training (Parallel)
- Train Classification/Regression models concurrently
- 3 models train simultaneously
- **Speedup**: ~3x faster than sequential

### Google API Integration
- Gemini AI for preprocessing decisions
- Fast inference with caching
- Error handling with fallback strategies

## 📈 Model Performance

The pipeline automatically selects the **best model** based on:

- **Classification**: Highest F1-Score with CV validation
- **Regression**: Highest R² Score with CV validation

All results include:
- Cross-validation metrics (mean ± std)
- Classification report (precision, recall, F1)
- Confusion matrix (for classification)
- RMSE, MAE, R² (for regression)

## 🔧 Configuration

### .env File
```
GOOGLE_API_KEY=your_api_key_here
```

### Customize Models

Edit `train_model.py` to change which models are trained:

```python
# Classification models
models_to_train = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]

# Regression models
models_to_train = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]
```

## 📝 Example Workflow

```bash
# 1. Start server
uvicorn api:app --reload

# 2. Upload CSV in browser (http://localhost:8000/docs)
# OR use curl

# 3. Server processes:
#    ✓ Loads data
#    ✓ Runs parallel EDA (2 tasks)
#    ✓ Calls Google Gemini for preprocessing plan
#    ✓ Applies preprocessing
#    ✓ Trains 3+ models in parallel
#    ✓ Selects best model
#    ✓ Saves model to disk
#    ✓ Returns results JSON

# 4. Get the model file path from response
# 5. Use for predictions!
```

## 🎓 How It Works

The new unified v2.0 pipeline streamlines the entire ML workflow:

1. **Load Data** — Parse CSV and detect task type
2. **Parallel EDA** — Run statistical analysis concurrently
3. **AI Preprocessing** — Gemini generates smart preprocessing plan
4. **Apply Pipeline** — Drop, impute, encode, scale automatically
5. **Parallel Training** — Train multiple models simultaneously
6. **Select Best** — Return top-performing model with metrics

**Why Parallel?**
- EDA tasks are independent ✓
- Model training is CPU-bound but can use multi-threading ✓
- Overall pipeline speedup: **~30-50% faster** than sequential

## ⚠️ Requirements

- Python 3.8+
- Google Gemini API key (free tier available)
- 2GB+ RAM recommended
- Internet connection (for Google API calls)

## 📄 License

MIT

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
