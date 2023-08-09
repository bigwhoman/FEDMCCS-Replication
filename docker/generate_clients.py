import os
import psutil

RESOURCES = [
    # In format of client id as the index of resource
    # -> (cpu cores, cpu core util in range of (0, 1], memory limit in MB,
    #     ping latency, bandwidth)
    # Use zero for ping and bandwidth to set them to unlimited
    (1, 0.5, 100, 100, 0),
    # (1, 1, 200, 0, 1024 * 1024),
    (2, 0.75, 500, 0, 0),
    (1, 0.25, 50, 50, 1024 * 512)
]
PROXY_PORT_START = 8088
SERVER_PORT = 8080
current_cpu_counter = 0

def validate_resources():
    # Check utilization
    if not all(map(lambda client: client[1] > 0 and client[1] <= 1, RESOURCES)):
        print("Invalid CPU utilization")
        exit(1)
    # Check CPU count
    available_cpus = os.cpu_count()
    declared_cpus = sum(map(lambda client: client[0], RESOURCES))
    if available_cpus < declared_cpus:
        print(f"You have declared {declared_cpus} CPUs but you only have {available_cpus}")
        exit(1)
    # Memory
    total_memory = psutil.virtual_memory().total // 1024 // 1024
    declared_memory = sum(map(lambda client: client[1], RESOURCES))
    if total_memory < declared_memory:
        print(f"You have declared {declared_memory} memory usage but you only have {total_memory}")
        exit(1)
    
def get_cpu_ids(needed: int) -> str:
    global current_cpu_counter
    result = ",".join(map(lambda i: str(i), range(current_cpu_counter, current_cpu_counter + needed)))
    current_cpu_counter += needed
    return result

def generate_resources(cpu_cores: int, cpu_util: float, memory_mb: int) -> str:
    return f"--cpuset-cpus '{get_cpu_ids(cpu_cores)}' --cpus '{float(cpu_cores) * cpu_util}' --memory '{memory_mb}M'"

def generate_compose_file():
    with open('runner.sh', 'w') as runner:
        runner.write("#!/bin/bash\n")
        runner.write("docker build --tag 'fed-client' .\n")
        runner.write(f"docker rm client{{1..{len(RESOURCES)}}}\n")
        for i, client in enumerate(RESOURCES):
            runner.write(f"tmux new-session -d -s 'client{i+1}-proxy' '../proxy-meter/proxy-meter --listen 0.0.0.0:{PROXY_PORT_START+i} --forward 127.0.0.1:{SERVER_PORT} --ping {client[3]}ms --speed {client[4]} &> proxy{i+1}.txt'\n")
            runner.write(f"docker run -d --name 'client{i+1}' --env PORT={PROXY_PORT_START+i} --env FREQUENCY={client[1]} --add-host=host.docker.internal:host-gateway {generate_resources(client[0], client[1], client[2])} fed-client\n")

validate_resources()
generate_compose_file()  # generate docker-compose file with num_clients clients
