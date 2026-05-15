# 只读浏览 SQLite（trend live_monitor / order_management、多腿 multi_leg 库）
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir sqlite-web==0.6.6
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "sqlite_web"]
