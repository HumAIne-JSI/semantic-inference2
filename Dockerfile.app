FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.app.txt ./
RUN pip install --no-cache-dir -r requirements.app.txt

COPY app.py ./

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV API_HOST=http://api:5001

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
