#!/bin/bash
docker build --tag 'fed-client' .
docker rm client{1..4}
docker run -d --name 'client1' --add-host=host.docker.internal:host-gateway --cpuset-cpus '0' --cpus '0.5' --memory '100M' fed-client
docker run -d --name 'client2' --add-host=host.docker.internal:host-gateway --cpuset-cpus '1' --cpus '1.0' --memory '200M' fed-client
docker run -d --name 'client3' --add-host=host.docker.internal:host-gateway --cpuset-cpus '2,3' --cpus '1.5' --memory '500M' fed-client
docker run -d --name 'client4' --add-host=host.docker.internal:host-gateway --cpuset-cpus '4' --cpus '0.25' --memory '50M' fed-client
