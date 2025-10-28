# DEAP Solver Family - Docker Image
# Stage 6.3 Implementation with Foundation Compliance

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    graphviz \
    graphviz-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create directories
RUN mkdir -p /app/data /app/logs /app/outputs /app/error_reports /tmp/numba_cache

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Install the package in editable mode
RUN pip install -e /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash deap_user && \
    chown -R deap_user:deap_user /app /tmp/numba_cache

# Switch to non-root user
USER deap_user

# Set up Python path
ENV PYTHONPATH="/app:/app/deap_family:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import deap; import numpy; import sympy; print('OK')" || exit 1

# Default command runs the __main__.py directly
CMD ["python", "/app/__main__.py", "--help"]

# Labels for metadata
LABEL maintainer="LUMEN Team [TEAM-ID: 93912]"
LABEL version="1.0.0"
LABEL description="DEAP Solver Family for Timetable Scheduling Optimization"
LABEL foundation_compliance="101%"

# Expose ports (if needed for monitoring)
EXPOSE 8080

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/logs", "/app/outputs", "/app/error_reports"]
