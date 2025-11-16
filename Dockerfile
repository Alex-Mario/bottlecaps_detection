# Mulai dari base image python 3.11-slim
FROM python:3.11-slim

WORKDIR /app

# Copy file-file penting
COPY pyproject.toml .
COPY configs/ /app/configs/
COPY src/ /app/src/

# Install dependensi dari pyproject.toml
# Ini akan otomatis meng-install ultralytics, typer, dll.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Entrypoint Anda sekarang adalah perintah `bsort`
ENTRYPOINT ["bsort"]