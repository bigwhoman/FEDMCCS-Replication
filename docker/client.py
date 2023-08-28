import warnings
from collections import OrderedDict
import os
import psutil
import time
import threading
import random

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

# Dynamic dataset
MAX_DATASET_SIZE = 60000 # MNIST data
ADDITION_RATE = 2000
current_dataset_size = MAX_DATASET_SIZE / int(os.environ['TOTAL_CLIENTS']) * random.uniform(0.9, 1.1)
print("Selected", current_dataset_size, "as dataset size")

# An array to keep the metrics of each trained data.
# Data stored is (freq, mem, time (seconds), dataset size)
epoch_metrics = []

class StatLoggerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cput_util_sum = 0
        self.memory_sum = 0
        self.sampled = 0
        self.done = False
    
    def run(self):
        while not self.done:
            self.sampled += 1
            self.memory_sum += psutil.Process().memory_info().rss
            self.cput_util_sum += float(os.environ['FREQUENCY'])
            time.sleep(1)


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
    # Increase a round counter
    global round_counter
    round_counter += 1
    # Create watcher thread
    stat_watcher = StatLoggerThread()
    stat_watcher.start()
    start_time = time.time()
    # Do the learning
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
    # Join the watcher thread
    end_time = time.time()
    stat_watcher.done = True
    stat_watcher.join()
    # Collect data
    return (stat_watcher.cput_util_sum / stat_watcher.sampled, stat_watcher.memory_sum / stat_watcher.sampled, end_time - start_time, len(trainloader.dataset))


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


def load_dataset(size: int):
    """Load MNIST training set with specified size"""
    # https://stackoverflow.com/a/55760170
    trainset = MNIST("./data", train=True, download=False, transform=ToTensor())
    trainset = torch.utils.data.random_split(trainset, [size, len(trainset)-size])[0]
    return DataLoader(trainset, batch_size=32, shuffle=True)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
TEST_LOADER = DataLoader(MNIST("./data", train=False, download=True, transform=ToTensor()))

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
        result["last_round"] = len(epoch_metrics)
        if len(epoch_metrics) != 0:
            result["last_round_freq"] = epoch_metrics[-1][0]
            result["last_round_mem"] = epoch_metrics[-1][1]
            result["last_round_time"] = epoch_metrics[-1][2]
            result["last_round_dataset_size"] = epoch_metrics[-1][3]
        result["freq"] = float(os.environ['FREQUENCY'])
        result["mem"] = int(os.environ['MEMORY']) * 1024 * 1024
        result["cores"] = int(os.environ['CORES'])
        result["dataset_size"] = current_dataset_size
        return result

    def fit(self, parameters, config):
        global current_dataset_size
        self.set_parameters(parameters)
        # Load the dataset of desired size
        dataset_loader = load_dataset(current_dataset_size)
        train_metrics = train(net, dataset_loader, epochs=1)
        # Save metrics
        epoch_metrics.append(train_metrics)
        print("Train", len(epoch_metrics), "done with params", epoch_metrics)
        # Change the dataset size and resample
        current_dataset_size = min(MAX_DATASET_SIZE, current_dataset_size + ADDITION_RATE * (random.random() / 0.5 + 0.9))
        print("Selected", current_dataset_size, "as dataset size")
        return self.get_parameters(config={}), len(dataset_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, TEST_LOADER)
        return loss, len(TEST_LOADER.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="host.docker.internal:" + os.environ['PORT'],
    client=FlowerClient(),
)
