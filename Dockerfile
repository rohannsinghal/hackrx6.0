# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy only the requirements file first to leverage Docker's build cache
COPY ./requirements.txt /code/requirements.txt

# Install all Python dependencies, without using a cache to ensure a fresh install
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt [cite: 1]

# Create writable directories for caches, databases, and temporary files
# and set the appropriate environment variables.
RUN mkdir -p /code/cache && chmod 777 /code/cache [cite: 2]
RUN mkdir -p /code/app/chroma_db && chmod -R 777 /code/app/chroma_db [cite: 2]
RUN mkdir -p /tmp/docs && chmod 777 /tmp/docs [cite: 2]
ENV HF_HOME=/code/cache 
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache 

# Now, copy the rest of your application code
COPY ./app /code/app 

# Define the command to run your application
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"] [cite: 1]