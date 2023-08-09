from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
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
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        print("penis")
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
        sampled_cids = []
        for cid in available_cids : 
            if cid %2 == 0 :
                print(f"cock {cid}")
                sampled_cids.append(cid)
        result = [self.clients[cid] for cid in sampled_cids]
        print("cock2 ", result)
        return result
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    client_manager= None,
    strategy=strategy,
)
