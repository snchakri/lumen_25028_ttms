# Stage 6.1 PuLP Solver Family - Docker Environment
# Installs all 5 solvers with mathematical rigor

FROM python:3.9-slim

# Install system dependencies for solvers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    coinor-cbc \
    coinor-clp \
    coinor-libcbc-dev \
    coinor-libclp-dev \
    glpk-utils \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

# Install HiGHS solver
RUN wget https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.4.2.tar.gz \
    && tar -xzf v1.4.2.tar.gz \
    && cd HiGHS-1.4.2 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd ../.. \
    && rm -rf HiGHS-1.4.2 v1.4.2.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PULP_CBC_PATH=/usr/bin/cbc
ENV PULP_GLPK_PATH=/usr/bin/glpsol

# Create directories for outputs
RUN mkdir -p /app/test_outputs /app/test_logs /app/test_errors

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]


