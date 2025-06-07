# LifeLens Quick Start Guide

## 🚀 How to Run LifeLens

Follow these simple steps to get the LifeLens Predictive Health and Survival Insight System running on your machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended
- Internet connection (for downloading ML models)

### Option 1: Quick Start (Recommended)

1. **Run the startup script:**
   ```bash
   python run_lifelens.py
   ```

   This script will:
   - Check your Python environment
   - Install dependencies if needed
   - Start the FastAPI server
   - Show you all the important URLs

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Step 3: Run the Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Accessing the System

Once the server is running, you can access:

- **📖 API Documentation**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/health
- **📈 Metrics**: http://localhost:8000/metrics
- **🏠 API Root**: http://localhost:8000/

## 🧪 Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Model Status
```bash
curl http://localhost:8000/api/v1/models/status
```

### 3. Survival Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/survival" \
     -H "Content-Type: application/json" \
     -d '{
       "health_record": {
         "age": 45,
         "gender": "M",
         "race": "White",
         "height_cm": 175,
         "weight_kg": 80,
         "blood_pressure_systolic": 120,
         "blood_pressure_diastolic": 80,
         "cholesterol": 200,
         "glucose": 90,
         "heart_rate": 70,
         "smoking": false,
         "diabetes": false,
         "hypertension": false,
         "family_history_heart": false,
         "exercise_hours_week": 3,
         "alcohol_drinks_week": 2
       },
       "prediction_horizon_years": 10
     }'
```

### 4. Health Event Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/events" \
     -H "Content-Type: application/json" \
     -d '{
       "health_record": {
         "age": 55,
         "gender": "F",
         "height_cm": 165,
         "weight_kg": 70,
         "blood_pressure_systolic": 140,
         "blood_pressure_diastolic": 90,
         "cholesterol": 240,
         "glucose": 110,
         "heart_rate": 75,
         "smoking": true,
         "diabetes": false,
         "hypertension": true,
         "family_history_heart": true,
         "exercise_hours_week": 1,
         "alcohol_drinks_week": 5
       }
     }'
```

### 5. Fairness Report
```bash
curl http://localhost:8000/api/v1/fairness/report
```

## 🔧 Troubleshooting

### Common Issues

1. **Port 8000 already in use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   # Or use a different port
   uvicorn app.main:app --port 8001
   ```

2. **Module not found errors**
   ```bash
   # Make sure you're in the backend directory
   cd backend
   # Set Python path
   export PYTHONPATH=$PWD
   ```

3. **Permission denied (macOS/Linux)**
   ```bash
   chmod +x run_lifelens.py
   python run_lifelens.py
   ```

4. **Memory issues**
   - The system uses ML models that require sufficient RAM
   - Close other applications if needed
   - Consider using a smaller model configuration

### Dependency Issues

If you encounter package conflicts:

```bash
# Clean install
pip uninstall -y -r backend/requirements.txt
pip install -r backend/requirements.txt

# Or use conda
conda create -n lifelens python=3.9
conda activate lifelens
pip install -r backend/requirements.txt
```

## 📱 Using the Interactive API Documentation

1. Go to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out" button
4. Fill in the required parameters
5. Click "Execute" to test the API

## 🏥 Sample Health Records for Testing

### Low Risk Patient
```json
{
  "age": 25,
  "gender": "F",
  "height_cm": 165,
  "weight_kg": 60,
  "blood_pressure_systolic": 110,
  "blood_pressure_diastolic": 70,
  "cholesterol": 180,
  "glucose": 85,
  "heart_rate": 65,
  "smoking": false,
  "diabetes": false,
  "hypertension": false,
  "family_history_heart": false,
  "exercise_hours_week": 5,
  "alcohol_drinks_week": 1
}
```

### High Risk Patient
```json
{
  "age": 65,
  "gender": "M",
  "height_cm": 170,
  "weight_kg": 100,
  "blood_pressure_systolic": 160,
  "blood_pressure_diastolic": 100,
  "cholesterol": 280,
  "glucose": 140,
  "heart_rate": 85,
  "smoking": true,
  "diabetes": true,
  "hypertension": true,
  "family_history_heart": true,
  "exercise_hours_week": 0,
  "alcohol_drinks_week": 10
}
```

## 🔄 Development Mode

For development with auto-reload:

```bash
cd backend
uvicorn app.main:app --reload --log-level debug
```

## 📊 Monitoring and Logs

- Logs are printed to console with structured formatting
- Prometheus metrics available at `/metrics`
- Health checks at `/health`
- Application metrics include request counts and response times

## 🆘 Getting Help

If you encounter issues:

1. Check the console logs for error messages
2. Verify all dependencies are installed correctly
3. Ensure you have sufficient system resources
4. Check the troubleshooting section above
5. Review the comprehensive README.md for more details

## 🎯 Next Steps

After getting the system running:

1. Explore the API documentation at `/docs`
2. Test different health scenarios
3. Monitor fairness reports
4. Review prediction explanations
5. Check system metrics

The LifeLens system is now ready for health prediction analysis! 🏥✨ 