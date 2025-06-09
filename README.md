
# LifeLens: Predictive Health & Survival Insight System

A full-stack ML platform that predicts survival probabilities and critical health events using structured medical data, clinical notes, and wearable sensor inputs.

[MIT License](https://opensource.org/licenses/MIT) • Python 3.9+ • FastAPI • React

---

## Overview

LifeLens combines machine learning and ethical AI to deliver accurate, interpretable health insights. It supports multi-modal data and provides a production-ready API with real-time inference and monitoring.

---

## Features

* **Multi-modal input**: EHR, clinical text (NLP), wearable data
* **Survival prediction**: 5 & 10-year risk estimates
* **Health event detection**: Cardiac arrest, stroke, diabetes (AUC: 0.67–1.00)
* **Time-series analysis**: LSTM/Transformer-based
* **Explainable AI**: SHAP, LIME, survival curves
* **Ethical AI**: Bias checks, fairness auditing
* **FastAPI backend** with logging, monitoring, and OpenAPI docs
* **React frontend** with real-time visualization

---

## Architecture

```
LifeLens/
├── backend/      # FastAPI server (models, API, data pipeline)
├── frontend/     # React client
├── models/       # Trained ML models
├── data/         # Input datasets
├── docker/       # Container setup
├── notebooks/    # Exploratory analysis
└── tests/        # Unit/integration tests
```

---

## Setup

### Prerequisites

* Python 3.9+, Node.js 16+, Docker (optional)

### Installation

```bash
git clone https://github.com/amukta14/LifeLens.git
cd LifeLens
```

**Backend**

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm start
```

**Or use Docker**

```bash
docker-compose up --build
```

---

## API Docs

* Swagger: `http://localhost:8001/docs`
* ReDoc: `http://localhost:8001/redoc`
* Health check: `http://localhost:8001/health`

**Key Endpoints**

* `POST /predict/survival`
* `POST /predict/events`
* `GET /explain/{prediction_id}`
* `POST /feedback`

---

## ML Pipeline

1. **Preprocessing**: Imputation, scaling, time alignment
2. **Feature Engineering**: Stats, embeddings (BERT), interaction terms
3. **Models**: XGBoost (tabular), LSTM/Transformer (temporal), BERT (text)
4. **Ensemble**: Meta-model for final prediction
5. **Explainability**: SHAP, LIME, visual outputs

---

## Ethical AI

* Bias monitoring (age, gender)
* Demographic parity checks
* User feedback integration and retraining

---

## Data Sources

* **MIMIC-IV** (critical care)
* Synthetic patient profiles
* Simulated wearable data (HRV, steps, sleep)

---

## Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

---

## Deployment

```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up --build
```

---

## Performance

* Survival C-index: > 0.85
* Event AUC: up to 1.00
* API latency: < 100ms
* SHAP explainability coverage: > 95%

---

## Contributing

1. Fork the repo
2. Create a branch
3. Commit & push
4. Open a pull request

---

## License

MIT – See `LICENSE`

---

## Credits

Thanks to MIMIC-IV, open-source contributors, and the healthcare AI community.

