FROM python:3.10-slim

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy package source under /app/or_tools_family so relative imports work
COPY . /app/or_tools_family

# Default entrypoint expects four args: stage3_output_path, output_path, log_path, error_report_path
ENTRYPOINT ["python", "-u", "-m", "or_tools_family.main"]
