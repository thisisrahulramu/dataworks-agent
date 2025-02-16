# Use the official slim Python 3.12 image based on Bookworm
FROM python:3.12-slim-bookworm

# Install system dependencies: curl, ca-certificates, git, Node.js, and npm.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Install Tesseract for OCR if easyocr requires it
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Set the PATH for uv (installed by uv installer below)
ENV PATH="/root/.local/bin:$PATH"

# (Optional) Install uv using the provided installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Command to run the application with uvicorn (or uv if preferred)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
