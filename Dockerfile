## Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

RUN mkdir -p data
# RUN touch myfile.brw

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 4567 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

# Copy the start.sh script and make it executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["sh", "/app/start.sh"]