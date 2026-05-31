# 🎨 Adobe MLOps Pipeline — Image Classifier

> **Project 1** from the Adobe Machine Learning Engineer (2026 Batch) application.  
> A production-grade MLOps system demonstrating model lifecycle management, CI/CD, A/B testing, drift detection, and observability.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                      │
│   push → lint/test → Docker build → ECR push → Deploy          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
 │  FastAPI    │   │   MLflow     │   │    Airflow       │
 │  (A/B test) │   │  (registry)  │   │  (drift DAG)     │
 │  port 8000  │   │  port 5000   │   │  port 8080       │
 └──────┬──────┘   └──────────────┘   └──────────────────┘
        │
        ▼
 ┌─────────────────────────────────┐
 │  Prometheus (9090)              │
 │  Pushgateway (9091)             │
 │  Grafana Dashboard (3000)       │
 └─────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML Framework** | PyTorch, EfficientNet-B0, Scikit-learn |
| **MLOps** | MLflow (experiment tracking, model registry) |
| **Orchestration** | Apache Airflow (drift-triggered retraining DAG) |
| **Serving** | FastAPI, Uvicorn |
| **Containerisation** | Docker, Docker Compose |
| **Cloud (AWS)** | SageMaker, ECR, S3 — via Terraform |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus, Grafana, Evidently AI |
| **Drift Detection** | Evidently AI (DataDriftPreset) |

---

## 🚀 Quick Start (Local — 5 minutes)

### Prerequisites
- Docker & Docker Compose installed
- 8 GB RAM recommended

### 1. Clone & start all services

```bash
git clone <your-repo-url>
cd mlops_project
docker-compose up --build -d
```

### 2. Wait for services (~2 min), then verify

```bash
docker-compose ps          # all should show "healthy"
curl http://localhost:8000/health
```

### 3. Open the UIs

| Service | URL | Credentials |
|---|---|---|
| **FastAPI docs** | http://localhost:8000/docs | — |
| **MLflow UI** | http://localhost:5000 | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |

---

## 🧪 Train a Model

```bash
# Option A: Smoke test with synthetic data (no dataset needed)
docker-compose exec api python model/train.py --epochs 2 --run_name smoke_test

# Option B: Real data (ImageFolder layout: data/images/<class>/*.jpg)
docker-compose exec api python model/train.py \
  --data_dir /app/data/images \
  --epochs 10 \
  --run_name baseline_v1
```

After training:
1. Open **MLflow UI** → Models → `image_classifier`
2. Assign alias **`stable`** to the best version
3. Restart the API: `docker-compose restart api`

---

## 🎯 Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict (with a public image URL)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png", "user_id": "user_001"}'

# Force a specific model version (for A/B testing QA)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://...", "user_id": "user_001", "force_version": "canary"}'

# Trigger rollback (shift all traffic to stable)
curl -X POST http://localhost:8000/rollback
```

---

## 📊 Run Drift Detection

```bash
# Run manually
docker-compose exec api python monitoring/drift_detector.py

# Check results
cat monitoring/latest_drift_results.json

# Open generated HTML report
open monitoring/reports/drift_report_*.html
```

The **Airflow DAG** (`mlops_image_classifier_pipeline`) runs this hourly and auto-triggers retraining if:
- >30% of features have drifted, **or**
- Model accuracy drops below 80%

---

## 🧪 Run Tests

```bash
# Run all unit tests
docker-compose exec api pytest tests/ -v

# With coverage
docker-compose exec api pytest tests/ -v --cov=app --cov=monitoring
```

---

## ☁️ AWS Deployment

```bash
cd infra

# 1. Configure AWS credentials
aws configure

# 2. Deploy infrastructure
terraform init
terraform plan -var="environment=dev"
terraform apply -var="environment=dev"

# 3. Push Docker image to ECR (output from terraform)
ECR_URL=$(terraform output -raw ecr_repo_url)
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker build -t $ECR_URL:stable .
docker push $ECR_URL:stable
```

---

## 📁 Project Structure

```
mlops_project/
├── app/
│   ├── main.py              # FastAPI app — predict, health, metrics, rollback
│   └── model_manager.py     # Model loading, A/B routing, inference
├── model/
│   └── train.py             # EfficientNet training with MLflow tracking
├── monitoring/
│   └── drift_detector.py    # Evidently-based drift analysis
├── pipeline/
│   └── mlops_dag.py         # Airflow DAG — drift → retrain → promote
├── tests/
│   └── test_api.py          # Unit tests (FastAPI + ModelManager)
├── infra/
│   ├── main.tf              # Terraform — ECR, S3, SageMaker, IAM
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana/             # Auto-provisioned Grafana dashboard
├── notebooks/
│   └── exploration.ipynb    # EDA, drift visualisation, A/B analysis
├── .github/workflows/
│   └── ci_cd.yml            # GitHub Actions — test → build → deploy
├── docker-compose.yml       # Full local stack
├── Dockerfile               # API container
└── requirements.txt
```

---

## 🎤 Interview Talking Points (Adobe-specific)

1. **Model lifecycle**: MLflow registry with `stable`/`canary` aliases enables zero-downtime promotion and instant rollback — same pattern Adobe uses for Firefly model updates.

2. **A/B testing**: Deterministic per-user hash routing means users get consistent experiences across requests (no flipping), while the split is configurable at runtime.

3. **CI/CD**: GitHub Actions pipeline with OIDC auth (no long-lived secrets), ECR image scanning on push, blue/green SageMaker deployment, automated smoke tests.

4. **Drift monitoring**: Evidently detects covariate shift (data drift) and concept drift independently. Airflow auto-retrains when thresholds are breached — closing the ML feedback loop.

5. **Governance**: All model versions tracked in MLflow with params, metrics, and artifacts. Rollback takes one API call. Audit trail is automatic.

