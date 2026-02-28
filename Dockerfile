# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files
# and to prevent stdout buffering (useful for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by XGBoost and LightGBM
# libgomp1 is the equivalent of libomp on Linux/Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies and a production WSGI server (Gunicorn)
# We use --no-cache-dir and --prefer-binary to prevent Render from running out of RAM (OOM) during build
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    gunicorn \
    flask-socketio==5.3.7 \
    eventlet==0.36.1

# Copy the rest of the application code and pre-trained model artifacts
COPY . .

# Expose the standard Flask port
EXPOSE 5000

# Set environment variable to indicate production environment
ENV FLASK_ENV=production

# Command to run the application using Gunicorn (production server)
# Enabled --capture-output and --error-logfile so we can see why it crashes in Render logs
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--threads", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "--capture-output", "--error-logfile", "-", "app:app"]
