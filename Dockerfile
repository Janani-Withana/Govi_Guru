# ---- build stage ----
FROM python:3.10-slim

# system deps for FAISS + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev git && rm -rf /var/lib/apt/lists/*

# workdir
WORKDIR /app

# requirements first (better cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project code
COPY app app

# serve through gunicorn (flask built-in server is not prod-ready)
ENV FLASK_APP=app
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]

