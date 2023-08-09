#!/bin/bash
docker build --tag 'fed-client' .
docker rm client{1..3}
tmux new-session -d -s 'client1-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8088 --forward 127.0.0.1:8080 --ping 100ms --speed 0 &> proxy1.txt'
docker run -d --name 'client1' --env PORT=8088 --add-host=host.docker.internal:host-gateway --cpuset-cpus '0' --cpus '0.5' --memory '500M' fed-client
tmux new-session -d -s 'client2-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8089 --forward 127.0.0.1:8080 --ping 0ms --speed 0 &> proxy2.txt'
docker run -d --name 'client2' --env PORT=8089 --add-host=host.docker.internal:host-gateway --cpuset-cpus '1,2' --cpus '1.5' --memory '500M' fed-client
tmux new-session -d -s 'client3-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8090 --forward 127.0.0.1:8080 --ping 50ms --speed 524288 &> proxy3.txt'
docker run -d --name 'client3' --env PORT=8090 --add-host=host.docker.internal:host-gateway --cpuset-cpus '3' --cpus '0.25' --memory '50M' fed-client
