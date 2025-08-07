# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- START: FINAL PERMISSION FIX ---
# Create a writable cache directory by changing its permissions.
# This allows any user (including the one running the app) to write to it.
RUN mkdir -p /code/cache && chmod 777 /code/cache
# Tell all Hugging Face libraries to use this new directory
ENV HF_HOME=/code/cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache
# --- END: FINAL PERMISSION FIX ---

# Copy your application code into the container
COPY ./app /code/app

# Command to run the application when the container launches
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]