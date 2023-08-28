import warnings
from collections import OrderedDict
import os
import psutil
import time
import threading

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# Hashmap to store metrics of this client.
# Unix timestamp to (frequency, memory)
metrics: dict[int, tuple[float, float]] = {}

class StatLoggerThread(threading.Thread):
    def run(self):
        while True:
            current_time = int(time.time())
            cpu_util = float(psutil.cpu_freq().current)
            memory = float(psutil.Process(os.getpid()).memory_info().rss)
            metrics[current_time] = (cpu_util, memory)
            print("Logged", (cpu_util, memory), "at", current_time)
            time.sleep(1)

# Get average of status of this client from 
def average_time_of_stat(start: int, end: int) -> (float, float):
    freq_sum = 0
    mem_sum = 0
    count = 0
    for i in range(start, end+1):
        if i in metrics:
            count += 1
            (freq, mem) = metrics[i]
            freq_sum += freq
            mem_sum += mem
    return (freq_sum / count), (mem_sum / count)


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512, bias=True)
        self.linear2 = torch.nn.Linear(512, 512, bias=True)
        self.linear3 = torch.nn.Linear(512, 10, bias=True)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, input):
        # Input is the picture so we flatten it just like before
        input = input.view(-1, 784)
        layer1 = nn.functional.relu(self.linear1(input))
        layer1 = self.dropout(layer1)
        layer2 = nn.functional.relu(self.linear2(layer1))
        layer2 = self.dropout(layer2)
        layer3 = self.linear3(layer2)
        return layer3


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = ToTensor()
    trainset = MNIST("./data", train=True, download=True, transform=trf)
    testset = MNIST("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def get_properties(*args, **kwargs):
        config = kwargs['config']
        print("props called with", config)
        result = {}
        if config["type"] == "total": # Device info
            result["cpu"] = int(os.environ['CORES'])
            result["frequency"] = int(os.environ['FREQUENCY'])
            result["memory"] = int(os.environ['MEMORY']) * 1024 * 1024
            result["ping"] = int(os.environ['PING'])
            result["speed"] = int(os.environ['BANDWIDTH'])
        else: # Get average utilization
            start = int(config["start"])
            end = int(config["end"])
            (freq, mem) = average_time_of_stat(start, end)
            result["frequency"] = freq
            result["memory"] = mem
        return result

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start utilization thread
StatLoggerThread().start()

# Start Flower client
fl.client.start_numpy_client(
    server_address="host.docker.internal:" + os.environ['PORT'],
    client=FlowerClient(),
)
