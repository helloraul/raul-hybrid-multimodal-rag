# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 libjpeg62-turbo libpng16-16 \
    libxml2 libxslt1.1 poppler-utils gcc build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "src/main.py", "--serve"]
