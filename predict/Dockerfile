# Base Image
FROM python:3.9-slim

# Set the working directory in the container
# WORKDIR /app
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install uvicorn
RUN pip install uvicorn

# Copy the rest of the application code to the working directory
COPY . .

# Expose a port for the application (if running a web server or API)
# EXPOSE 8080  # Uncomment if you need to expose a port

# Set environment variables (optional)
# ENV ENVIRONMENT=production

# Run any additional setup commands (optional)
# RUN python setup.py install

# Command to run the application (default)
# CMD ["python", "process_and_train.py"]

# Alternatively, for a more complex application, you might start a web server
EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
