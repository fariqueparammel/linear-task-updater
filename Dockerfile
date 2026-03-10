FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY config.py models.py state.py main.py ./
COPY agents/ ./agents/

# State and logs persist via volumes
RUN mkdir -p /app/state /app/logs

VOLUME ["/app/state", "/app/logs"]

# Non-root user for security
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-u", "main.py"]
