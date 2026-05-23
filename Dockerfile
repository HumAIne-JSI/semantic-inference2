# Use Python 3.11 for broader package wheel compatibility in deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
# Copy all Python modules to avoid missing-import/runtime issues
# and to prevent build failures when optional files are absent.
COPY *.py .
COPY README.md .
COPY *.json .
COPY *.txt .
COPY *.yaml .

# Expose port for Flask API
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=semantic_search_api.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "semantic_search_api.py"]
