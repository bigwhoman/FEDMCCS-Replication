import yaml
import sys

def generate_compose_file(num_clients):
    services = {}
    for i in range(1, num_clients + 1):
        services[f'client{i}'] = {
            'build': '.',
            'volumes': ['.:/client'],
            'extra_hosts': ['host.docker.internal:host-gateway']
        }
    compose_dict = {'version': '3.8', 'services': services}
    
    with open('docker-compose.yml', 'w') as file:
        yaml.dump(compose_dict, file, default_flow_style=False)

# Get number of clients from command line arguments
num_clients = int(sys.argv[1])
generate_compose_file(num_clients)  # generate docker-compose file with num_clients clients
