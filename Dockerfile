FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download cars_cleaned.csv from Google Drive at build time
RUN mkdir -p data && \
    curl -L "https://drive.google.com/uc?export=download&id=13N9KED54uFKTCGSZHJTAyPSLzsY27PuN" \
    -o data/cars_cleaned.csv

RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]