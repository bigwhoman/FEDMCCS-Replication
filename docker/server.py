from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import GetPropertiesIns
from flwr.common.logger import log
from logging import INFO
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from typing import Dict, List, Optional

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


class myClientManager(fl.server.SimpleClientManager):
    round_number = 0
    client_configs = {}


    def predict_utilization(client):  
        historical_data = client
        Util = {}
        for parameter in ["frequency","memory","time","cpu"] :
            model = LinearRegression()
            model.fit(historical_data["last_stat"]["size"], historical_data["last_stat"][parameter])
            y_pred = model.predict(client["current_dataset_size"])
            Util[parameter] = y_pred
        return Util




    def linear_regression(available_cids):
        pass

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        self.round_number += 1
        print(" round number ---------------------> ",self.round_number)
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return [] 
        

        keylist = ["frequency","memory","time","cpu","size"]

        for cid in available_cids:
            if cid not in self.client_configs : 
                self.client_configs[cid]["historical_data"] = {key:[] for key in keylist}
                self.client_configs[cid]["current_dataset_size"] = 0
            config = GetPropertiesIns({"resource": "total/old"})
            serv_conf = self.clients[cid].get_properties(config, None).properties
            for key in keylist :
                self.client_configs[cid]["historical_data"][key].append(serv_conf[key])
            self.client_configs[cid]["current_dataset_size"] = serv_conf["current_dataset_size"]
        if self.round_number <= 4 : 
            print(" selection -----------------> random...")
            sampled_cids = random.sample(available_cids, num_clients)
        else :
            print(" selection -----------------> learning...")
            sampled_cids = self.linear_regression(available_cids)
        print(num_clients)
        return [self.clients[cid] for cid in sampled_cids]
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    client_manager=myClientManager(),
    strategy=strategy,
)
