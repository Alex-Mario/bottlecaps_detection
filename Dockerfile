# Base image
FROM python:3.11-slim

# Set working directory di container
WORKDIR /app

# Copy semua file dari repo ke container
COPY . .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set entrypoint agar bisa langsung pakai CLI bsort
ENTRYPOINT ["python", "-m", "bsort.cli"]
