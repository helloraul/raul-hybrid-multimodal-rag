FROM python:3.11-slim

# system deps you actually use (example: poppler-utils for pdftotext, tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
