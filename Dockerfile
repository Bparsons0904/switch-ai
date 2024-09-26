FROM python:3.12
WORKDIR /app
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
RUN echo "Contents of /app:" && ls -R /app
CMD ["sh", "-c", "python -c 'import sys; print(sys.path)' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app"]
