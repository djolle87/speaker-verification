FROM python:3.10-slim

# Install system dependencies (includes libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory to the project root
WORKDIR /app

# Copy everything to /app in the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run main.py from root
CMD ["python", "main.py"]
