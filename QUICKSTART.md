# 🚀 QUICK START GUIDE

## What I Created for You

✅ **api.py** - Simple FastAPI server (no complex code!)
✅ **requirements.txt** - All dependencies listed
✅ **Updated README.md** - With API and next steps

---

## ⚡ Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Your API Key
Make sure `.env` file has your Google Gemini API key:
```
GOOGLE_API_KEY=your_key_here
```

### Step 3: Start the API Server
```bash
python api.py
```

You should see:
```
[INFO] Starting FastAPI server...
[INFO] Open browser at http://localhost:8000/docs for interactive API docs
```

---

## 🌐 Use the API

Open **http://localhost:8000/docs** in your browser to see:
- Interactive documentation
- All endpoints
- Try API calls directly
- See responses in real-time

---

## 📡 Three Ways to Use the Pipeline

### Option 1: Via Browser/Swagger UI (Easiest)
- Go to http://localhost:8000/docs
- Click on `/predict` endpoint
- Upload a CSV file
- See results instantly!

### Option 2: Via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_data.csv"
```

### Option 3: Via Python
```python
import requests

with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    print(response.json())
```

---

## 📂 API Endpoints

| Endpoint | What It Does |
|----------|-------------|
| `GET /` | Welcome & info |
| `GET /health` | Check if server is running |
| `GET /files` | List CSV files in csv_data/ |
| `POST /predict` | Upload CSV → Run pipeline |
| `POST /predict-path` | Use existing CSV file |

---

## 🎯 What Happens When You Run Pipeline

1. **Uploads** your CSV
2. **Analyzes** it with Gemini AI → Statistical report
3. **Preprocesses** it with AI → Cleaned data
4. **Trains** multiple ML models → Performance scores
5. **Returns** all results as JSON

---

## ✅ Next Things You Can Do

1. **Test with sample data:**
   - You already have `student_performance.csv` in csv_data/
   - Try: http://localhost:8000/predict-path?file_name=student_performance.csv

2. **Add your own CSV:**
   - Drop any CSV in `csv_data/` folder
   - Use the API to process it

3. **Customize:**
   - Modify `api.py` to add your own endpoints
   - Change preprocessing logic in `data_processing.py`
   - Add new models in `train_model.py`

4. **Deploy:**
   - Use Uvicorn with proper config
   - Deploy to cloud (Heroku, Railway, AWS, etc.)

---

## 🆘 Troubleshooting

**Port 8000 already in use?**
```bash
python api.py --port 8001
```

**Missing dependencies?**
```bash
pip install --upgrade -r requirements.txt
```

**API key error?**
- Check `.env` file exists
- Verify API key from https://makersuite.google.com/app/apikey

---

## 📖 Learn More

- Full docs in `README.md`
- FastAPI docs: https://fastapi.tiangolo.com/
- All code is well-commented!

---

**That's it! Your API is ready to go! 🎉**
