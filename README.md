# 🎨 Adobe MLOps Image Classifier

> **Built for Adobe Machine Learning Engineer (2026 Batch) Application**  
> A production-grade MLOps pipeline that trains, deploys, monitors, and auto-retrains an image classification model — mirroring Adobe's real infrastructure for Firefly and Creative Cloud AI features.

[![CI/CD](https://github.com/sumith25-dev/ML-ops-Image-Classification/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/sumith25-dev/ML-ops-Image-Classification/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.13.0-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## 🏆 What I Built

A complete end-to-end MLOps system with:

- **Image Classification API** — EfficientNet-B0 model trained on 14,000 real images achieving **86.85% validation accuracy**
- **A/B Testing** — Deterministic per-user routing (80% stable / 20% canary) with instant rollback
- **MLflow Tracking** — Full experiment tracking, model registry, and versioning
- **Drift Detection** — Automatic data drift monitoring with Evidently AI
- **CI/CD Pipeline** — GitHub Actions with lint, test, and Docker build stages ✅ Passing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                      │
│         push → lint → test → Docker build → deploy         │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
 ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
 │  FastAPI    │ │   MLflow     │ │   PostgreSQL     │
 │  A/B Test   │ │  Registry    │ │   Database       │
 │  port 8000  │ │  port 5001   │ │   port 5432      │
 └─────────────┘ └──────────────┘ └──────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML Framework** | PyTorch 2.3, EfficientNet-B0, Scikit-learn |
| **MLOps** | MLflow 2.13 (experiment tracking, model registry) |
| **Serving** | FastAPI, Uvicorn |
| **Containerisation** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions ✅ |
| **Drift Detection** | Evidently AI |
| **Database** | PostgreSQL |
| **Cloud Ready** | Terraform (AWS ECR, S3, SageMaker) |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Model** | EfficientNet-B0 (ImageNet pretrained) |
| **Dataset** | Intel Image Classification (14,000 real images) |
| **Classes** | buildings, forest, glacier, mountain, sea, street |
| **Training Epochs** | 5 |
| **Best Val Accuracy** | **86.85%** |
| **Model Version** | v2 (registered in MLflow) |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- 4GB free disk space

### Run locally
```bash
git clone https://github.com/sumith25-dev/ML-ops-Image-Classification.git
cd ML-ops-Image-Classification
docker-compose up -d api mlflow postgres
```

### Open the UIs

| Service | URL |
|---|---|
| **FastAPI docs** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5001 |

---

## 🧪 Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.pexels.com/photos/1547813/pexels-photo-1547813.jpeg",
    "user_id": "user_001"
  }'
```

Response:
```json
{
  "label": "forest",
  "confidence": 0.87,
  "model_version": "stable",
  "latency_ms": 245.32
}
```

---

## 🎯 Key Features

### 1. A/B Testing
```
User ID → MD5 Hash → Bucket (0-9999)
  0-7999  → stable model  (80% traffic)
  8000-9999 → canary model (20% traffic)
```
Same user always hits the same model — consistent experience.

### 2. Model Lifecycle
```
Train → MLflow Registry → Assign 'stable' alias → API loads automatically
                       → Assign 'canary' alias  → 20% traffic routed
```

### 3. Drift Detection
```
Run drift_detector.py
→ If drift > 30% OR accuracy < 80%
→ Flag for retraining
→ Promote new model to canary
```

### 4. CI/CD Pipeline
```
git push → GitHub Actions
         → Lint (ruff)      ✅
         → Unit Tests       ✅
         → Docker Build     ✅
```

---

## 📁 Project Structure

```
ML-ops-Image-Classification/
├── app/
│   ├── main.py              # FastAPI — predict, health, metrics, rollback
│   └── model_manager.py     # MLflow loading, A/B routing, inference
├── model/
│   └── train.py             # EfficientNet training with MLflow tracking
├── monitoring/
│   └── drift_detector.py    # Evidently drift analysis
├── pipeline/
│   └── mlops_dag.py         # Airflow DAG — drift → retrain → promote
├── tests/
│   └── test_api.py          # Unit tests
├── infra/
│   ├── main.tf              # Terraform — AWS ECR, S3, SageMaker
│   └── prometheus.yml       # Prometheus config (future deployment)
├── notebooks/
│   └── exploration.ipynb    # EDA and drift visualisation
├── .github/workflows/
│   └── ci_cd.yml            # GitHub Actions CI/CD ✅ passing
├── docker-compose.yml       # Full local stack
├── Dockerfile               # API container
└── requirements.txt
```

---

## 🌊 Drift Detection

```python
# Run manually or schedule via Airflow
python monitoring/drift_detector.py

# Auto-retraining triggers when:
# - drift_share >= 30% of features drifted
# - model accuracy < 80%
```

---

## ☁️ AWS Deployment (Ready)

```bash
cd infra
terraform init
terraform apply -var="environment=prod"
```

Provisions:
- ECR repository for Docker images
- S3 bucket for MLflow artifacts
- SageMaker endpoint with blue/green deployment
- IAM roles and policies

---

Here  click my [@project-website] (https://ml-ops-image-classification-production.up.railway.app/docs#/default/predict_predict_post)
I trained of 5 epochs, if it trained around 10 epochs then it confidence will increase from 20 to 80-90 per

1. **86.85% accuracy** on 14,000 real images using EfficientNet-B0

2. **A/B testing** — Deterministic per-user hash routing for consistent user experience, configurable split at runtime

3. **CI/CD** — Every commit automatically lints, tests, and builds Docker image via GitHub Actions ✅

4. **Drift detection** — Evidently AI detects when production data drifts from training distribution, triggering automatic retraining

5. **MLflow registry** — Complete model versioning with aliases, metrics tracking, and artifact storage

6. **Zero-downtime deployment** — Blue/green deployment with instant rollback via single API call

---

## 👨‍💻 Author

**Sumit** — Applying for Adobe Machine Learning Engineer (2026 Batch)

GitHub: [@sumith25-dev](https://github.com/sumith25-dev)

---

## 📄 License

MIT License
