FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model inside the container at build time
# This guarantees pkl files are saved with the same Python/joblib version that loads them
RUN python src/train.py

RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]