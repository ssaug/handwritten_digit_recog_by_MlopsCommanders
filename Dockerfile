# Use the relevant Python Image from Dockerhub
FROM python:3.12.4-slim

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .

# Install from requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the API
CMD ["python", "app.py"]