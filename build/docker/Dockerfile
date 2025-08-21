# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create directories for the application
RUN mkdir -p /app/data/images/input /app/data/images/annotated /app/src/libs/sam_libs/models

# Copy requirements first to leverage Docker cache
COPY ../../config/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from project root
COPY ../../ .

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set ownership of the app directory
RUN chown -R appuser:appuser /app

# Create volume mount points
VOLUME ["/app/data/images", "/app/src/libs/sam_libs/models"]

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/aisegmentation || exit 1

# Default command
CMD ["python", "app_secure.py"]
