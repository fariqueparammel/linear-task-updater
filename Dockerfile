FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gosu for dropping privileges after fixing permissions
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY config.py models.py state.py main.py ./
COPY agents/ ./agents/
COPY entrypoint.sh ./

# State and logs persist via volumes
RUN mkdir -p /app/state /app/logs

VOLUME ["/app/state", "/app/logs"]

# Create non-root user (entrypoint fixes volume perms then drops to this user)
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

ENTRYPOINT ["./entrypoint.sh"]
