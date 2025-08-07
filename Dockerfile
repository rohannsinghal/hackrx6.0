# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- START: FINAL PERMISSION FIX ---
# Create a writable cache directory AND set the correct owner for the app user
RUN mkdir /code/cache && chown -R user:user /code/cache
# Tell all Hugging Face libraries to use this new directory (the modern way)
ENV HF_HOME=/code/cache
# Also set the specific variable for sentence-transformers for good measure
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache
# --- END: FINAL PERMISSION FIX ---

# Copy your application code into the container
COPY ./app /code/app

# Command to run the application when the container launches
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]