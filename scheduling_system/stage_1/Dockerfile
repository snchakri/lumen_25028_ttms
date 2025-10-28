# Stage-1 Input Validation Testing Environment
# Self-contained Docker setup for isolated testing

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-xdist>=3.3.0

# Copy entire stage_1 directory
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app

# Create test directories
RUN mkdir -p /app/test_data
RUN mkdir -p /app/test_outputs
RUN mkdir -p /app/test_logs

# Default command - run tests
CMD ["bash", "-c", "cd /app && python test_stage1_direct.py"]



