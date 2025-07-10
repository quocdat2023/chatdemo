# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for MongoDB and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up a cache directory with proper permissions
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache/huggingface

# Copy the entire application code into the container
COPY . .

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Command to run the application with Flask-SocketIO
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]