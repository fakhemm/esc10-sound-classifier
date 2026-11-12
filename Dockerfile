# Builds an image that serves the trained model through the API in app.py

# Start from a small official Python image (the same Python version used for development)
FROM python:3.13-slim

# Work inside the folder /app in the container
WORKDIR /app

# Install the Python packages first so that this slow step is reused when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained model
COPY app.py features.py model.keras ./

# The API listens on port 8000
EXPOSE 8000

# Start the server (--host 0.0.0.0 makes it reachable from outside the container)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
