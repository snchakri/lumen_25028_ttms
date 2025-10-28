# Dockerfile for PyGMO Solver Family (Stage 6.4)
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder - Compile PyGMO and dependencies
# ============================================================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    libnlopt-dev \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Install runtime dependencies only (Debian trixie versions)
RUN apt-get update && apt-get install -y \
    libboost-serialization1.83.0 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libgsl28 \
    libnlopt0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:$PYTHONPATH"

# Create app user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/pygmo_family /data/input /data/output /data/logs && \
    chown -R appuser:appuser /app /data

# Set working directory
WORKDIR /app

# Copy application code to pygmo_family subdirectory
COPY --chown=appuser:appuser . /app/pygmo_family/

# Switch to non-root user
USER appuser

# Create necessary directories
RUN mkdir -p \
    /data/input/L_raw \
    /data/input/L_rel \
    /data/input/L_idx \
    /data/input/L_opt \
    /data/input/metadata \
    /data/output \
    /data/logs \
    /data/checkpoints

# Expose port for potential future API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pygmo; print('PyGMO OK')" || exit 1

# Default command: Run main solver
CMD ["python", "-m", "pygmo_family", \
     "--input-dir", "/data/input", \
     "--output-dir", "/data/output", \
     "--log-dir", "/data/logs", \
     "--solver", "NSGA-II"]

# ============================================================================
# Build instructions:
#   docker build -t pygmo-solver:latest -f Dockerfile .
#
# Run instructions:
#   docker run -v $(pwd)/test_data:/data/input \
#              -v $(pwd)/test_output:/data/output \
#              -v $(pwd)/test_logs:/data/logs \
#              pygmo-solver:latest
#
# Interactive mode:
#   docker run -it --entrypoint /bin/bash pygmo-solver:latest
# ============================================================================
