# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- ADD THESE TWO LINES ---
# Create a writable directory for model caches
RUN mkdir /code/cache
# Tell the library to use this new directory
ENV TRANSFORMERS_CACHE=/code/cache
# --- END OF ADDITION ---

# Copy your application code into the container
COPY ./app /code/app

# Command to run the application when the container launches
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]