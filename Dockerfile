# 베이스 이미지 (슬림)
FROM python:3.11-slim

# (추가) LightGBM이 의존하는 OpenMP 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# 파이썬 런타임 기본 옵션
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 먼저 복사 → 캐시 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# (선택) 로컬 테스트용
EXPOSE 8000

# EB가 주입하는 $PORT 사용 (gunicorn + uvicorn worker)
CMD ["sh","-c","gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:${PORT:-8000} --timeout 120 --graceful-timeout 30 --keep-alive 5 --access-logfile - app.main:app"]
