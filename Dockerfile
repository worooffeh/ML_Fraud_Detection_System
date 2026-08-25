# ---------------------------------------------------------------------------
# Nova Pay fraud scoring API image
# ---------------------------------------------------------------------------
# This is the "what runs" guarantee. Everything the app needs — the exact
# Python version, the exact library versions, the model artifacts — is baked
# in here, so the container behaves identically on your laptop and on EC2.
#
# Build once, run anywhere. No fresh `pip install` on the server, so the
# os/httpx2/pandas-3.x problems from the deployment log simply can't recur.
# ---------------------------------------------------------------------------

# Pin the base image to an EXACT Python version (not just "3.11").
# This is the single most important line for reproducibility.

FROM python:3.11.9-slim

# Don't write .pyc files; stream logs straight to the console (so `docker logs`
# and journald see them immediately).

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# A few system libraries scikit-learn / scipy wheels expect at runtime.

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements FIRST and install them as their own layer.
# Docker caches this layer, so as long as requirements.txt is unchanged,
# rebuilds skip re-installing everything — fast iteration.

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the application code and the trained model artifacts.

COPY service/ ./service/

# data/ is only needed if the app reads CSVs at runtime; **usually not for serving**
# Create a non-root user to run the app — never run app processes as root.

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/service/nova_pay_fraud_model_lean.joblib

# Document the port the API listens on inside the container.
EXPOSE 8000

# A container-level health check. Docker/ECS will mark the container unhealthy
# if /health stops responding — this pairs with the "make /health real" item
# on your hardening checklist.

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/ready').status==200 else sys.exit(1)"

# Start the API. Bind to 0.0.0.0 INSIDE the container (that's the container's
# own network namespace, not the host) — we still only publish it to the
# host's 127.0.0.1 in the run command / compose file.

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
