# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- START: FINAL PERMISSION FIXES ---
# Create a writable directory for model caches
RUN mkdir -p /code/cache && chmod 777 /code/cache
# Create a writable directory for the vector database
RUN mkdir -p /code/app/chroma_db && chmod -R 777 /code/app/chroma_db
# Tell all Hugging Face libraries to use the new cache directory
ENV HF_HOME=/code/cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache
# --- END: FINAL PERMISSION FIXES ---

# Copy your application code into the container
COPY ./app /code/app

# Command to run the application when the container launches
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]