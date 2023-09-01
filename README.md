# FEDMCCS-Replication
Replicating FEDMCCS client selection in Federated Learning
<br> 
## The paper
The paper is in the link : [FEDMCCS](https://ieeexplore.ieee.org/document/9212434)


## Running

At first run the `python3 generate_clients.py` then run the `runner.sh` with bash or sh.

If you get status code 137 it's likely caused by OOM killer. See `/var/log/kern.log`.

## Server

The server runs on 0.0.0.0:8080 on the linux server.<br>
Once the first clinet connects to the server, it starts the procedure. The overall server code is similar to  [Flower Base Server](https://github.com/adap/flower/blob/main/src/py/flwr/server/server.py) with changes to clientManager. 
<br>
`myClientManager` class is a class which inherits `fl.server.SimpleClientManager`. We need to override the `sample` method defined in the parent class and used for client selection at the beginnig of each <b>evaluation</b> and <b>training round</b>. 
<br>
The method, first check if there are any clients available and if there were enough clients, it would proceed to select the clients according to a criteria. <br>
In the first few rounds, all of the clients are given a round so that the server could have enough data for the linear regression.
`RANDOM_ROUNDS` is a variable which determines how much rounds all the clients are trained so we have a fair amount of data. <br>
After some rounds, clients are selected according to the linear regression method. It gets a list of all available clinets and first checks their historical data in which there are 4 parameters : <br>
<li> Used Cores</li>
<li> Used Frequencies</li>
<li> Used Memories</li>
<li> Training Times</li>
<li> Used Dataset Sizes</li>

`predict utilization` predicts the next rounds <b>Cores, Frequencies, Memories and Training Times</b> according to the <b>dataset size</b> which is going to be used.<br>
If the wanted number of clients are selected according to `CLIENT FRACTION`, there will be no training. <br>
After the clients are selected, we might have a problem : <b>Less clients are selected because of the budget limit</b>. So to mitigate this impact, if less clients are selected in the linear regression, the remainder of selected clients are randomly selected. This ensures convergence.

## Test Setup

### HPC Server
<li> Ubuntu 22.04</li>
<li> Linux Kernel 5.15</li>
<li> Docker 24.0.5</li>
<li> CPU : AMD EPYC 7763 64-Core Processor</li>
<li> CPU Cores : 12</li>

### Client Configurations : 
| **Client**   | **CPU Cores** | **Cpu Utilization** | **Memory Limit(MB)** | **Ping Latency(ms)** | **Bandwidth (Mbps)** |
|----------|-----------|-----------------|------------------|------------------|------------------|
| Client1  | 1         | 0.01            | 250              | 100              | 512              |
| Client2  | 1         | 0.01            | 150              | 100              | 512              |
| Client3  | 1         | 1               | 200              | no latency       | 1024             |
| Client4  | 1         | 0.01            | 150              | 100              | 512              |
| Client5  | 1         | 1               | 100              | 100              | 512              |
| Client6  | 1         | 1               | 50               | 50               | 512              |
| Client7  | 1         | 0.01            | 100              | 100              | 512              |
| Client8  | 1         | 0.8             | No Limit         | no latency       | 1024             |
| Client9  | 1         | 0.01            | 100              | 100              | 512              |
| Client10 | 1         | 1               | 100              | 100              | 512              |
| Client11 | 1         | 0.01            | 100              | 100              | 512              |

### Resource Budget
| **Resource**          	| **Budget** 	|
|-------------------	|--------	|
| Memory Budget     	| 0.8    	|
| CPU Budget        	| 1      	|
| Energy Budget     	| 1      	|
| Time Threshold(s) 	| 20     	|

### Other Parameters
| **Parameter**                      	|     	|
|------------------------------------	|-----	|
| Client Fraction                    	| 0.5 	|
| Number of Rounds                   	| 15  	|
| Random Rounds  (Before Regression) 	| 5   	|
| Dataset                             | MNIST |
| Model                               | CNN   |

## Test results
### Round Duration
![Round Duration](https://github.com/bigwhoman/FEDMCCS-Replication/assets/79264715/6093db09-1974-4312-a492-7d6b28b5e945)
| Round        | FEDMCCS | Random Select |
|---------|---------|---------------|
| round1  | 19.2    | 13.5          |
| round2  | 49.4    | 43            |
| round3  | 79.6    | 72.2          |
| round4  | 109.7   | 101.5         |
| round5  | 139.7   | 130.8         |
| round6  | 169.6   | 160           |
| round7  | 191.5   | 189.3         |
| round8  | 203.3   | 214.3         |
| round9  | 219.3   | 229.9         |
| round10 | 241.2   | 259.1         |
| round11 | 356.4   | 373.6         |
| round12 | 1329.4  | 1519.6        |
| round13 | 2268.4  | 2662.6        |
| round14 | 3183.4  | 3889.6        |
| round15 | 4194.4  | 5120.6        |

### Convergence
![FEDMCCS and Random](https://github.com/bigwhoman/FEDMCCS-Replication/assets/79264715/f581f986-d3f5-4365-94ab-30d2425c9b02)

![FEDMCCS and Random (1)](https://github.com/bigwhoman/FEDMCCS-Replication/assets/79264715/2679838e-e66a-4ef9-9f31-e30bf3481b9d)


| Time   | FEDMCCS |   | Time   | Random |
|--------|---------|---|--------|--------|
|   19.2 |    0.33 |   |   13.5 |   0.25 |
|   49.4 |    0.48 |   |     43 |   0.44 |
|   79.6 |    0.56 |   |   72.2 |   0.58 |
|  109.7 |    0.63 |   |  101.5 |   0.65 |
|  139.7 |     0.7 |   |  130.8 |   0.72 |
|  169.6 |    0.75 |   |    160 |   0.76 |
|  191.5 |    0.79 |   |  189.3 |    0.8 |
|  203.3 |    0.81 |   |  214.3 |   0.82 |
|  219.3 |    0.83 |   |  229.9 |   0.84 |
|  241.2 |    0.84 |   |  259.1 |   0.85 |
|  356.4 |    0.85 |   |  373.6 |   0.85 |
| 1329.4 |    0.87 |   | 1519.6 |   0.86 |
| 2268.4 |   0.868 |   | 2662.6 |   0.87 |
| 3183.4 |   0.873 |   | 3889.6 |   0.87 |
| 4194.4 |   0.877 |   | 5120.6 |   0.87 |



## Conclusions

### Advantages

### Disadvantages
