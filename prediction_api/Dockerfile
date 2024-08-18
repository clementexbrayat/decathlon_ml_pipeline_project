# Base Image
FROM python:3.9-slim

# Set the working directory in the container
# WORKDIR /app
WORKDIR /app

# Copy the requirements file to the working directory
COPY prediction_api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code to the working directory
COPY . .

# Set the entry point for this container
ENTRYPOINT ["python", "prediction_api/app.py"]
