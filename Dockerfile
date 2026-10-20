FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY scripts ./scripts
COPY dashboard ./dashboard

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# The dashboard is static and is served by the API process at /dashboard/.
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
