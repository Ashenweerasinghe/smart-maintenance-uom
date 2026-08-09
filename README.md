# 🛠️ Smart Predictive Maintenance Platform

An end-to-end AI application combining **classical Machine Learning** and **Generative AI**, deployed to **Microsoft Azure** with **no Docker** (direct code deployment).

**🌐 Live app:** https://smart-maintenance-uom.azurewebsites.net
**📘 Interactive API docs (Swagger):** https://smart-maintenance-uom.azurewebsites.net/docs

> Submitted for the *Cloud Computing for Artificial Intelligence* module — MSc in Artificial Intelligence, University of Moratuwa.

---

## 1. Problem Statement

Unplanned equipment failure is one of the largest sources of cost in manufacturing. When a machine fails without warning it causes production downtime, wasted material, safety hazards, and expensive emergency repairs. Reliability engineers need a way to (a) **anticipate failures before they happen** from live sensor data, and (b) **quickly reason about what to do** when a machine looks unhealthy.

## 2. Use Case

The platform serves industrial **reliability / maintenance engineers** and covers **two complementary AI use cases** in a single application:

1. **Failure Prediction (Predictive/Classification):** Given real-time sensor readings (temperatures, rotational speed, torque, tool wear) the app predicts whether a machine is likely to fail, with a probability and risk level. This enables *proactive* maintenance scheduling instead of reactive repairs.
2. **Generative-AI Maintenance Assistant (Chatbot):** A domain-focused conversational assistant that answers troubleshooting and preventive-maintenance questions **and explains the prediction results in plain language**, recommending concrete next steps.

Where can it be used? On a plant-floor dashboard, integrated into a CMMS (Computerised Maintenance Management System), or called as a REST API by other monitoring systems.

## 3. Solution Overview

The two AI capabilities reinforce each other. The ML model produces a **quantitative** risk signal; the Gen-AI assistant turns that signal into **actionable, human-readable** guidance. A user can enter sensor values, get an instant failure prediction, then click **“Explain this”** to have the assistant interpret the result and suggest checks — one coherent workflow.

The whole thing is exposed as **both a web UI and a REST API** from a single FastAPI service, deployed directly to Azure App Service.

## 4. Dataset

- **Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository (id = 601). Publicly available.
- **Size:** 10,000 rows (synthetic but realistic industrial data).
- **Features used:** `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`, and `Type` (product quality: L/M/H).
- **Target:** `Machine failure` (binary: 0 = healthy, 1 = failure). The dataset is imbalanced — only **3.4%** of rows are failures.
- The dataset is fetched via the `ucimlrepo` package in [`model/train.py`](model/train.py) and cached to [`data/ai4i2020.csv`](data/ai4i2020.csv).

## 5. AI/ML Approach

### Failure Predictor (classical ML)
- **Model:** `RandomForestClassifier` (300 trees, `class_weight="balanced"`) inside a scikit-learn `Pipeline`.
- **Preprocessing:** `StandardScaler` on numeric features + `OneHotEncoder` on the categorical `Type`, wired through a `ColumnTransformer` so the exact same transforms run at train and inference time.
- **Class imbalance:** Because failures are rare, a default 0.5 threshold under-detects them. We **tune the decision threshold** on the test set to maximise F1 on the failure class, landing at **0.28**. The tuned threshold is stored in [`model/metadata.json`](model/metadata.json) and applied by the API.
- **Evaluation (held-out 20% test set):**

  | Metric | Value |
  |---|---|
  | ROC-AUC | **0.963** |
  | Accuracy | 0.982 |
  | Precision (failure) | 0.75 |
  | Recall (failure) | 0.71 |
  | F1 (failure) | 0.73 |

  Threshold tuning raised failure **recall from 0.46 → 0.71** — the app now catches ~71% of true failures, the right trade-off when a missed failure is far costlier than a false alarm.

### Maintenance Assistant (Generative AI)
- **Model:** `gpt-4.1-mini` deployed in **Azure AI Foundry** (Azure OpenAI).
- **Technique:** system-prompted, domain-scoped chat completions with optional **prediction context injection** so the assistant can explain a specific prediction. Conversation history is passed for multi-turn context.
- Secrets (endpoint/key/deployment) are read from **environment variables** — never committed.

## 6. Application Architecture

```
                         ┌───────────────────────────────────────────────┐
  Developer  ── push ──► │  GitHub repo + GitHub Actions (CI/CD)          │
                         └───────────────────┬───────────────────────────┘
                                             ▼  direct code deploy (no Docker, Oryx build)
  Evaluator              ┌───────────────────────────────────────────────┐
  browser  ────────────► │  Azure App Service (Linux · Python 3.11)       │
  (public URL)           │  FastAPI + Gunicorn/Uvicorn                     │
                         │   • GET  /            → Web UI (predict + chat) │
                         │   • POST /predict     → ML JSON API             │
                         │   • POST /chat        → Gen-AI JSON API         │
                         │   • GET  /docs        → Swagger UI              │
                         │   • loads model.pkl at startup                  │
                         └───────┬─────────────────────────┬──────────────┘
                                 │ local inference          │ HTTPS + key (env vars)
                                 ▼                          ▼
                        scikit-learn model       ┌────────────────────────────┐
                        (RandomForest .pkl)       │  Azure AI Foundry           │
                                                  │  gpt-4.1-mini deployment    │
                                                  └────────────────────────────┘
```

**Components**
- **`app/main.py`** — FastAPI app: routes, request/response schemas, web UI.
- **`app/ml.py`** — loads the trained pipeline + metadata, runs predictions with the tuned threshold.
- **`app/chat.py`** — Azure AI Foundry client for the Maintenance Assistant (graceful fallback if unconfigured).
- **`app/templates/` + `app/static/`** — the web front-end (form + chatbot).
- **`model/train.py`** — reproducible training script → `model.pkl` + `metadata.json`.

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, pandas, numpy, joblib |
| Web / API | FastAPI, Uvicorn, Gunicorn, Jinja2 |
| Generative AI | Azure AI Foundry (Azure OpenAI), `openai` SDK, model `gpt-4.1-mini` |
| Cloud | Azure App Service (Linux, Python) — direct code deploy, **no Docker** |
| CI/CD | GitHub Actions |
| Dataset access | `ucimlrepo` |

## 8. Local Setup Instructions

```bash
# 1. Clone
git clone https://github.com/Ashenweerasinghe/smart-maintenance-uom.git
cd smart-maintenance-uom

# 2. (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the model — a trained model.pkl is already committed
python model/train.py

# 5. (Optional) enable the Gen-AI assistant locally
cp .env.example .env        # then fill in your Azure AI Foundry values

# 6. Run the app
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

Without the `.env` values the app runs fine — the predictor works fully and the chatbot returns a friendly “not configured” message.

## 9. Deployment Details

Deployed to **Azure App Service** using **direct (non-container) deployment** — the platform builds the app from `requirements.txt` with Oryx, so no Docker image is required.

**Cloud resources (subscription: Visual Studio Enterprise / MSDN):**

| Resource | Name | Region |
|---|---|---|
| Resource group | `rg-smart-maintenance-uom` | — |
| App Service plan | `plan-smart-maintenance-uom` (Free F1, Linux) | Southeast Asia |
| Web App | `smart-maintenance-uom` | Southeast Asia |
| Azure OpenAI (Foundry) | `oai-smart-maintenance-uom` | East US 2 |
| Model deployment | `gpt-4.1-mini` (GlobalStandard) | East US 2 |

**Reproduce the deployment (Azure CLI):**

```bash
RG=rg-smart-maintenance-uom
az group create -n $RG -l eastus2

# Azure AI Foundry (Azure OpenAI) + model
az cognitiveservices account create -n oai-smart-maintenance-uom -g $RG \
  -l eastus2 --kind OpenAI --sku S0 --custom-domain oai-smart-maintenance-uom --yes
az cognitiveservices account deployment create -n oai-smart-maintenance-uom -g $RG \
  --deployment-name gpt-4.1-mini --model-name gpt-4.1-mini --model-version 2025-04-14 \
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity 30

# App Service (Linux, Python 3.11) — no Docker
az appservice plan create -n plan-smart-maintenance-uom -g $RG -l southeastasia --sku F1 --is-linux
az webapp create -n smart-maintenance-uom -g $RG --plan plan-smart-maintenance-uom --runtime "PYTHON:3.11"
az webapp config set -n smart-maintenance-uom -g $RG \
  --startup-file "gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set -n smart-maintenance-uom -g $RG --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000 \
  AZURE_OPENAI_ENDPOINT=<endpoint> AZURE_OPENAI_API_KEY=<key> \
  AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini AZURE_OPENAI_API_VERSION=2024-10-21

# Deploy code (zip → Oryx build)
az webapp deploy -n smart-maintenance-uom -g $RG --src-path deploy.zip --type zip
```

**CI/CD (optional):** A ready-to-use GitHub Actions workflow (`.github/workflows/azure-deploy.yml`) is included in the project. Once enabled, pushes to `main` auto-deploy to App Service using the publish profile stored as the `AZUREAPPSERVICE_PUBLISHPROFILE` repository secret. To enable it: grant the GitHub token `workflow` scope (`gh auth refresh -s workflow`), add the publish profile secret, and push the workflow file. The live deployment above was performed directly with `az webapp deploy`.

## 10. API / Web Application Usage

### Web UI
Open the [live app](https://smart-maintenance-uom.azurewebsites.net): fill the sensor form and click **Predict**, then click **🤖 Explain this** or chat directly with the Maintenance Assistant.

### REST API

**`POST /predict`**
```bash
curl -X POST https://smart-maintenance-uom.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":302.5,"process_temperature":311.0,"rotational_speed":1300,"torque":68.0,"tool_wear":230,"type":"L"}'
```
```json
{"prediction":1,"prediction_label":"Failure","failure_probability":0.91,"risk_level":"High","decision_threshold":0.28}
```

**`POST /chat`**
```bash
curl -X POST https://smart-maintenance-uom.azurewebsites.net/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tool wear is 230 min — what should I check?"}'
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/predict` | POST | Machine-failure prediction |
| `/chat` | POST | Gen-AI Maintenance Assistant |
| `/metadata` | GET | Model schema, feature ranges, metrics |
| `/health` | GET | Health probe |
| `/docs` | GET | Swagger / OpenAPI UI |

## 11. Docker Instructions

**Not applicable.** As permitted by the assignment, Docker is **not used** — the application is deployed directly to Azure App Service, which builds it from `requirements.txt` (Oryx). No Dockerfile or container registry is required. The startup command used by App Service is in [`startup.sh`](startup.sh).

---

### Repository layout
```
.
├── app/
│   ├── main.py            # FastAPI routes + schemas
│   ├── ml.py              # model loading + prediction
│   ├── chat.py            # Azure AI Foundry client
│   ├── templates/index.html
│   └── static/style.css
├── model/
│   ├── train.py           # training pipeline
│   ├── model.pkl          # trained model (committed)
│   └── metadata.json      # feature schema, threshold, metrics
├── data/ai4i2020.csv      # cached dataset
├── .github/workflows/azure-deploy.yml
├── requirements.txt
├── startup.sh
└── .env.example
```
