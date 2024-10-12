# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY requirement.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of your application code
COPY . .

# Command to run your application
CMD ["python", "your_script.py"]  
