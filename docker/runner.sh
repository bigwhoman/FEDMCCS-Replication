#!/bin/bash
docker build --tag 'fed-client' .
docker rm client{1..4}
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8088/tcp
fi
tmux kill-session -t 'client1-proxy'
tmux new-session -d -s 'client1-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8088 --forward 127.0.0.1:8080 --ping 100ms --speed 0 &> proxy1.txt'
docker run -d --name 'client1' --env PORT=8088 --env CORES=1 --env FREQUENCY=2500 --env MEMORY=100 --env PING=100 --env BANDWIDTH=0 --add-host=host.docker.internal:host-gateway --cpuset-cpus '0' --cpus '0.5' --memory '100M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8089/tcp
fi
tmux kill-session -t 'client2-proxy'
tmux new-session -d -s 'client2-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8089 --forward 127.0.0.1:8080 --ping 0ms --speed 1048576 &> proxy2.txt'
docker run -d --name 'client2' --env PORT=8089 --env CORES=1 --env FREQUENCY=5000 --env MEMORY=200 --env PING=0 --env BANDWIDTH=1048576 --add-host=host.docker.internal:host-gateway --cpuset-cpus '1' --cpus '1.0' --memory '200M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8090/tcp
fi
tmux kill-session -t 'client3-proxy'
tmux new-session -d -s 'client3-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8090 --forward 127.0.0.1:8080 --ping 0ms --speed 0 &> proxy3.txt'
docker run -d --name 'client3' --env PORT=8090 --env CORES=2 --env FREQUENCY=3750 --env MEMORY=500 --env PING=0 --env BANDWIDTH=0 --add-host=host.docker.internal:host-gateway --cpuset-cpus '2,3' --cpus '1.5' --memory '500M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8091/tcp
fi
tmux kill-session -t 'client4-proxy'
tmux new-session -d -s 'client4-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8091 --forward 127.0.0.1:8080 --ping 50ms --speed 524288 &> proxy4.txt'
docker run -d --name 'client4' --env PORT=8091 --env CORES=1 --env FREQUENCY=1250 --env MEMORY=50 --env PING=50 --env BANDWIDTH=524288 --add-host=host.docker.internal:host-gateway --cpuset-cpus '4' --cpus '0.25' --memory '50M' fed-client
