# Stage-2 Docker Image for Foundation-Compliant Student Batching
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Stage-2 source code
COPY . /app/stage_2/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command: run tests
CMD ["python", "-m", "pytest", "stage_2/tests/", "-v", "--tb=short", "--json-report", "--json-report-file=/app/test_results.json"]
