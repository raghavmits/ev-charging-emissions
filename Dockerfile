# Use the official Python image from DockerHub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for Streamlit
EXPOSE 8051

# Run the Streamlit app
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.headless=true"]

# To be tested later: CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
