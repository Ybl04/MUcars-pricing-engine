#!/bin/bash
# Start FastAPI in the background
cd /app/api && uvicorn main:app --host 0.0.0.0 --port 8000 &

# Give it 3 seconds to start before Streamlit tries to call it
sleep 3

# Start Streamlit on the port HuggingFace expects
cd /app/app && streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0