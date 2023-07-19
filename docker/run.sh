#!/bin/bash

# Prompt for the number of clients
echo "Enter the number of clients:"
read num_clients

# Generate the docker-compose file
python3 generate_clients.py $num_clients

# Build and run the docker-compose file
sudo docker-compose up --build
