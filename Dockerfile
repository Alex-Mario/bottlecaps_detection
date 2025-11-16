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
    # LANGKAH 1: Install torch & torchvision versi CPU-ONLY (SIZE KECIL)
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    # LANGKAH 2: Install sisa proyek kita
    # (pip akan melihat torch sudah ada & tidak akan download versi GPU)
    pip install --no-cache-dir .

# Entrypoint Anda sekarang adalah perintah `bsort`
ENTRYPOINT ["bsort"]