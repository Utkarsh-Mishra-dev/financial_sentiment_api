# Lightweight Python base image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer - Docker caches this,
# so re-builds are fast unless requirements.txt itself changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app (code + trained .pkl files)
COPY . .

# Render sets $PORT at runtime; default to 8000 for local testing
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
