# Stage 7: Output Validation - Docker Configuration
# For testing and deployment

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (kept minimal)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY scheduling_engine_localized/stage_7/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire scheduling_engine_localized package for proper imports
COPY scheduling_engine_localized /app/scheduling_engine_localized

# Set Python path to package root
ENV PYTHONPATH=/app

# Default command (help)
CMD ["python", "-m", "scheduling_engine_localized.stage_7.main", "--help"]
