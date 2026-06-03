# Build context: repository root (see docker-compose.yml)
FROM python:3.12-slim

WORKDIR /app

COPY deploy/business-console/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/__init__.py /app/src/__init__.py
COPY src/mlbot_console /app/src/mlbot_console
COPY src/live_data_stream /app/src/live_data_stream
COPY src/time_series_model /app/src/time_series_model
COPY src/monitoring /app/src/monitoring
# strategy_stage_regions → archetype.loader → src.config.strategy_layout
COPY src/config /app/src/config
COPY config/strategies /app/config/strategies
COPY config/monitoring /app/config/monitoring

ENV PYTHONPATH=/app/src
ENV MLBOT_CONSOLE_PORT=8800
ENV MLBOT_CONSOLE_STRATEGIES_ROOT=/app/config/strategies

EXPOSE 8800
CMD ["python", "-m", "uvicorn", "mlbot_console.main:app", "--host", "0.0.0.0", "--port", "8800"]
