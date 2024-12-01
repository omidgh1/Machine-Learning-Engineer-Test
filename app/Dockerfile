# Use Python 3.10 as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable to ensure unbuffered output
ENV PYTHONUNBUFFERED=1

EXPOSE 4000
# Command to run when the container spins up.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000", "--proxy-headers"]
