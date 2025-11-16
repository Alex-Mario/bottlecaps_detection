# Mulai dari base image Ultralytics, ini cara termudah
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Copy file-file penting
COPY pyproject.toml .
COPY configs/ /app/configs/
COPY src/ /app/src/

# Install aplikasi Anda (dan dependensinya) dari pyproject.toml
RUN pip install .

# Entrypoint Anda sekarang adalah perintah `bsort`
ENTRYPOINT ["bsort"]