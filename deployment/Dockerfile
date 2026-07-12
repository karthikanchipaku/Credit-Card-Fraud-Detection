FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Force install sqlalchemy in case it was missed
RUN pip install sqlalchemy

# Copy project files
COPY . .

# Set environment path to root
ENV PYTHONPATH=/app

# Add this as the last line
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]