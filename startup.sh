#!/bin/bash
# Startup command for Azure App Service (Linux, Python).
# Serves the FastAPI app with Gunicorn + Uvicorn workers.
gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:8000 \
  --timeout 600
