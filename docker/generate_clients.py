import yaml
import os
import psutil

RESOURCES = [
    # In format of client id as the index of resource
    # -> (cpu cores, cpu core util in range of (0, 1], memory limit in MB)
    (1, 0.5, 100),
    (1, 1, 200),
    (2, 0.75, 500),
    (1, 0.25, 50)
]
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

def generate_resources(cpu_cores: int, cpu_util: float, memory_mb: int):
    limits = {'cpuset-cpus': get_cpu_ids(cpu_cores), 'cpus': str(float(cpu_cores) * cpu_util), 'memory': f"{memory_mb}M"}
    return {'limits': limits}

def generate_compose_file():
    services = {}
    for i, client in enumerate(RESOURCES):
        services[f'client{i+1}'] = {
            'build': '.',
            'volumes': ['.:/client'],
            'extra_hosts': ['host.docker.internal:host-gateway'],
            'deploy': {'resources': generate_resources(client[0], client[1], client[2])}
        }
    compose_dict = {'version': '3.8', 'services': services}
    
    with open('docker-compose.yml', 'w') as file:
        yaml.dump(compose_dict, file, default_flow_style=False)

validate_resources()
generate_compose_file()  # generate docker-compose file with num_clients clients
