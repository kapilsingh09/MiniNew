# 🚀 Automated ML Pipeline

An AI-powered end-to-end machine learning pipeline that **analyzes**, **preprocesses**, and **trains models** on any CSV dataset — automatically.

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

## Setup

```bash
# Install dependencies
pip install pandas numpy scikit-learn google-genai python-dotenv

# Set your API key in .env
echo "GOOGLE_API_KEY=your_key_here" > .env

# Drop any CSV into csv_data/ folder, then run:
python main.py
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

## Requirements

- Python 3.8+
- Google Gemini API key
- pandas, numpy, scikit-learn, google-genai, python-dotenv
