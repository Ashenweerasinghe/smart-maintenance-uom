"""Smart Predictive Maintenance Platform - FastAPI application.

Exposes both a human web UI and a REST API:
    GET  /            web UI (prediction form + chatbot)
    POST /predict     machine-failure prediction (JSON)
    POST /chat        maintenance-assistant chatbot (JSON, Azure AI Foundry)
    GET  /health      health probe
    GET  /docs        auto-generated Swagger UI
"""
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from . import chat as chat_mod
from . import ml

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Smart Predictive Maintenance Platform",
    description=(
        "End-to-end AI app: a machine-failure predictor (scikit-learn) and a "
        "Gen-AI Maintenance Assistant (Azure AI Foundry)."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ----------------------------- Schemas -----------------------------------
class PredictRequest(BaseModel):
    air_temperature: float = Field(..., example=298.1, description="Air temperature [K]")
    process_temperature: float = Field(..., example=308.6, description="Process temperature [K]")
    rotational_speed: float = Field(..., example=1551, description="Rotational speed [rpm]")
    torque: float = Field(..., example=42.8, description="Torque [Nm]")
    tool_wear: float = Field(..., example=108, description="Tool wear [min]")
    type: str = Field("L", example="L", description="Product quality type: L, M, or H")


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    failure_probability: float
    risk_level: str
    decision_threshold: float


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., example="Torque is 65 Nm and tool wear 210 min - is that risky?")
    history: Optional[List[ChatTurn]] = None
    prediction_context: Optional[str] = Field(
        None, example="Failure probability 0.82, risk High"
    )


class ChatResponse(BaseModel):
    reply: str
    configured: bool


# ----------------------------- Routes ------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    meta = ml.get_metadata()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "meta": meta,
            "chat_configured": chat_mod.is_configured(),
        },
    )


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "chat_configured": chat_mod.is_configured()}


@app.get("/metadata", tags=["prediction"])
def metadata():
    """Model schema, feature ranges, and evaluation metrics."""
    return ml.get_metadata()


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    """Predict machine failure from sensor readings."""
    features: Dict[str, object] = {
        "Air temperature": req.air_temperature,
        "Process temperature": req.process_temperature,
        "Rotational speed": req.rotational_speed,
        "Torque": req.torque,
        "Tool wear": req.tool_wear,
        "Type": req.type,
    }
    return ml.predict(features)


@app.post("/chat", response_model=ChatResponse, tags=["assistant"])
def chat(req: ChatRequest):
    """Ask the Gen-AI Maintenance Assistant (Azure AI Foundry)."""
    history = [t.model_dump() for t in req.history] if req.history else None
    reply = chat_mod.chat(
        req.message, history=history, prediction_context=req.prediction_context
    )
    return {"reply": reply, "configured": chat_mod.is_configured()}
