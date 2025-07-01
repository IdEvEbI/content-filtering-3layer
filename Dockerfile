FROM python:3.12-alpine
WORKDIR /app

# 安装编译依赖（pyahocorasick 需要）
RUN apk add --no-cache build-base gcc musl-dev

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001
CMD ["uvicorn", "ac_service.main:app", "--host", "0.0.0.0", "--port", "8001"]