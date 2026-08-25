# Start from a small official Python image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first (this layer is cached for faster rebuilds)
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# Copy the service code and the model artifact into the image
COPY service/ ./service/

ENV MODEL_PATH=/app/service/nova_pay_fraud_model_lean.joblib \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 appuser \
	&& chown -R appuser:appuser /app
USER appuser

# Document the port the service listens on
EXPOSE 8000

# The command that runs when the container starts
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
	CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
