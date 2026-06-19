FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download cars_cleaned.csv from Google Drive at build time
RUN mkdir -p data && \
    curl -c /tmp/cookies.txt -s -L \
    "https://drive.google.com/uc?export=download&id=13N9KED54uFKTCGSZHJTAyPSLzsY27PuN" \
    | grep -o 'confirm=[^&"]*' | head -1 > /tmp/confirm.txt && \
    curl -L -b /tmp/cookies.txt \
    "https://drive.google.com/uc?export=download&confirm=$(cat /tmp/confirm.txt)&id=13N9KED54uFKTCGSZHJTAyPSLzsY27PuN" \
    -o data/cars_cleaned.csv && \
    echo "Download complete. File size:" && \
    wc -c data/cars_cleaned.csv

RUN python src/train.py

RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]