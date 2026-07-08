# Use official slim Python runtime
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for compiling python extensions if any
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list
COPY requirements.txt .

# Install dependencies. Use CPU version for torch to keep container size reasonable
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app/chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
