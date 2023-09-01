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
import time
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
    round_duration = 0
    round_start = 0
    round_end = 0
    ### TODO
    RANDOM_ROUNDS = 5
    CPU_BUDGET = 1 
    MEM_BUDGET = 0.8 
    ENERGY_BUDGET = 1
    TIME_THRESHOLD = 20
    CLIENT_FRACTION = 0.5
    ###

    def predict_utilization(
                            self, 
                            client):  # client = client_configs[cid]
        try : 
            historical_data = client["historical_data"]
            # print("predicttttttttttttt------------------------------------------------")
            Util = {}
            for parameter in ["last_round_freq","last_round_mem","last_round_time","last_round_cores"] :
                model = LinearRegression()
                x = np.array(historical_data["last_round_dataset_size"]).reshape((-1, 1))
                y = np.array(historical_data[parameter])

                # print("last data set --------------> ", historical_data["last_round_dataset_size"])
                # print(f"parammmmmmmmmmmmm {parameter} ------------> ", historical_data[parameter])

                model.fit(x, y) 
                y_pred = model.predict(np.array([client["dataset_size"]]).reshape((-1, 1)))
                Util[parameter] = y_pred[0]
            return Util
        except Exception as E :
            print("faiiiiiiiiled ---------> ", client)


    def sufficientResources(self, client) :
        Util = self.predict_utilization(client)
        # print(f"The util : {client}  ------------> ",Util)
        if Util['last_round_cores'] > self.CPU_BUDGET * client['cores'] : 
            print("no cores ---------> ",Util['last_round_cores'], self.CPU_BUDGET * client['cores'])
        if Util['last_round_freq'] > self.ENERGY_BUDGET * client['freq'] : 
            print("no freq ---------> ",Util['last_round_freq'], self.ENERGY_BUDGET * client['freq'])
        if Util['last_round_mem'] > self.MEM_BUDGET * client['mem'] : 
            print("no mem ---------> ",Util['last_round_mem'], self.MEM_BUDGET * client['mem'])
        if Util['last_round_time'] > self.TIME_THRESHOLD : 
            print("no time ---------> ",Util['last_round_time'], self.TIME_THRESHOLD)
        return (Util['last_round_cores'] <= self.CPU_BUDGET * client['cores'] and
                Util['last_round_mem'] <= self.MEM_BUDGET * client['mem'] and  
                Util['last_round_freq'] <= self.ENERGY_BUDGET * client['freq'] and
                Util['last_round_time'] <= self.TIME_THRESHOLD)  

    def linear_regression(
                            self, 
                            available_cids,
                            num_clients, 
                            target_fraction):
        selected_clients = []

        # while available_cids and len(selected_clients) < target_fraction * num_clients:
        print("available cids -----------------> ", random.sample(available_cids,len(available_cids)))
        for cid in random.sample(available_cids,len(available_cids)) : 
            if len(selected_clients) >= target_fraction * num_clients :
                break
            if len(self.client_configs[cid]["historical_data"]) == 0  \
                 or len(self.client_configs[cid]["historical_data"]["last_round_freq"]) == 0 :
                selected_clients.append(cid)
            elif self.sufficientResources(self.client_configs[cid]) :
                selected_clients.append(cid)
        print("selected clients -------------> ",selected_clients)
        return selected_clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        self.round_number += 1
        if self.round_number > 1 :
            self.MEM_BUDGET *= 1.05
            self.ENERGY_BUDGET *= 1.05
            self.TIME_THRESHOLD *= 1.05
            self.round_duration = time.time() - self.round_start
            print(f"round {self.round_number - 1} duration = {self.round_duration}") 

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
        

        keylist = ["last_round_freq","last_round_mem","last_round_time","last_round_cores","last_round_dataset_size"]

        for cid in available_cids:
            # print(f"my niggaaaaaaaaaaaaaa {cid}")
            config = GetPropertiesIns({"resource": "total/old"})
            serv_conf = self.clients[cid].get_properties(config, None).properties

            if cid not in self.client_configs : 
                self.client_configs[cid] = {}
                self.client_configs[cid]["historical_data"] = {key:[] for key in keylist}
                self.client_configs[cid]["mem"] = serv_conf["mem"]
                self.client_configs[cid]["freq"] = serv_conf["freq"]
                self.client_configs[cid]["cores"] = serv_conf["cores"]
                self.client_configs[cid]["dataset_size"] = 0        

            for key in keylist :
                    if key in serv_conf and key:
                        self.client_configs[cid]["historical_data"][key].append(serv_conf[key])
            # print("cliennnnnnnnnnnnnnnnnnnnnnnnntttttttt  configs --------------> ",self.client_configs[cid])
            self.client_configs[cid]["dataset_size"] = serv_conf["dataset_size"]

        sampled_cids = []
        if self.round_number <= self.RANDOM_ROUNDS * 2 : 
            print(" selection -----------------> random...")
            sampled_cids = random.sample(available_cids, num_clients)
    
        else :
            print(" selection -----------------> learning...")
            sampled_cids = self.linear_regression(available_cids,num_clients, self.CLIENT_FRACTION)
            if len(sampled_cids) < num_clients*self.CLIENT_FRACTION :
                print(f"ONLY {len(sampled_cids)} CLINET SELECTED BY LEARNING !!!!!!!")
             
             for cid in sampled_cids :
                 available_cids.remove(cid)
             others = random.sample(available_cids, int(num_clients*self.CLIENT_FRACTION) - len(sampled_cids))
             sampled_cids.extend(others)
        print("sampled cids ----------------> ", sampled_cids)
        self.round_start = time.time()
        return [self.clients[cid] for cid in sampled_cids]
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    client_manager=myClientManager(),
    strategy=strategy,
)
