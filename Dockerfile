#finalversion
# Use an official, lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file first to leverage Docker's build cache
COPY ./requirements.txt /code/requirements.txt

# Install all Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into a subdirectory
COPY ./app /code/app

# --- FIX 1: Set the working directory to where your app code is ---
WORKDIR /code/app

# --- FIX 2: CRITICAL - Expose the port your app runs on ---
# This tells Hugging Face where to send traffic.
EXPOSE 7860

# Define the command to run your application
# This now correctly runs from inside the /code/app directory
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "7860"]