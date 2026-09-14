# Lightweight Python 3.11 base image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose container port
EXPOSE 8501

# Streamlit Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch Streamlit with dynamic PORT and 0.0.0.0 binding
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
