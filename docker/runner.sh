#!/bin/bash
docker build --tag 'fed-client' .
docker rm client{1..6}
SEED=$RANDOM
echo Chose $SEED as the seed
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8088/tcp
fi
tmux kill-session -t 'client1-proxy'
tmux new-session -d -s 'client1-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8088 --forward 127.0.0.1:8080 --ping 100ms --speed 524288 &> proxy1.txt'
docker run -d --name 'client1' --env CLIENT_ID=0 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8088 --env CORES=1 --env FREQUENCY=5000 --env MEMORY=250 --env PING=100 --env BANDWIDTH=524288 --add-host=host.docker.internal:host-gateway --cpuset-cpus '0' --cpus '1.0' --memory '250M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8089/tcp
fi
tmux kill-session -t 'client2-proxy'
tmux new-session -d -s 'client2-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8089 --forward 127.0.0.1:8080 --ping 100ms --speed 524288 &> proxy2.txt'
docker run -d --name 'client2' --env CLIENT_ID=1 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8089 --env CORES=1 --env FREQUENCY=5000 --env MEMORY=150 --env PING=100 --env BANDWIDTH=524288 --add-host=host.docker.internal:host-gateway --cpuset-cpus '1' --cpus '1.0' --memory '150M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8090/tcp
fi
tmux kill-session -t 'client3-proxy'
tmux new-session -d -s 'client3-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8090 --forward 127.0.0.1:8080 --ping 100ms --speed 524288 &> proxy3.txt'
docker run -d --name 'client3' --env CLIENT_ID=2 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8090 --env CORES=1 --env FREQUENCY=5000 --env MEMORY=150 --env PING=100 --env BANDWIDTH=524288 --add-host=host.docker.internal:host-gateway --cpuset-cpus '2' --cpus '1.0' --memory '150M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8091/tcp
fi
tmux kill-session -t 'client4-proxy'
tmux new-session -d -s 'client4-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8091 --forward 127.0.0.1:8080 --ping 0ms --speed 1048576 &> proxy4.txt'
docker run -d --name 'client4' --env CLIENT_ID=3 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8091 --env CORES=1 --env FREQUENCY=5000 --env MEMORY=200 --env PING=0 --env BANDWIDTH=1048576 --add-host=host.docker.internal:host-gateway --cpuset-cpus '3' --cpus '1.0' --memory '200M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8092/tcp
fi
tmux kill-session -t 'client5-proxy'
tmux new-session -d -s 'client5-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8092 --forward 127.0.0.1:8080 --ping 0ms --speed 1048576 &> proxy5.txt'
docker run -d --name 'client5' --env CLIENT_ID=4 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8092 --env CORES=2 --env FREQUENCY=3750 --env MEMORY=500 --env PING=0 --env BANDWIDTH=1048576 --add-host=host.docker.internal:host-gateway --cpuset-cpus '4,5' --cpus '1.5' --memory '500M' fed-client
if ! [[ "$(sudo ufw status)" =~ "inactive" ]]; then
	sudo ufw allow 8093/tcp
fi
tmux kill-session -t 'client6-proxy'
tmux new-session -d -s 'client6-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:8093 --forward 127.0.0.1:8080 --ping 50ms --speed 524288 &> proxy6.txt'
docker run -d --name 'client6' --env CLIENT_ID=5 --env TOTAL_CLIENTS=6 --env SEED=$SEED --env PORT=8093 --env CORES=1 --env FREQUENCY=1250 --env MEMORY=150 --env PING=50 --env BANDWIDTH=524288 --add-host=host.docker.internal:host-gateway --cpuset-cpus '6' --cpus '0.25' --memory '150M' fed-client
