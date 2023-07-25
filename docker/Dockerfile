# Use the official Python image from the Docker Hub
FROM python:3.8-slim-buster

# Install the Flower library
RUN pip install flwr

# Install torch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Add the client code to the /client directory in the container
ADD client.py /client/

# Set /client as the working directory
WORKDIR /client/

EXPOSE 8080

CMD ["python", "client.py"]
