# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set the working directory to /app
WORKDIR /usr/src/app

# Install system packages that may be required
RUN apt-get update && apt-get install -y build-essential

# Copy the current directory contents into the container at /app
COPY src/ /usr/src/app
COPY requirements.txt /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run your application
CMD ["flask", "run"]