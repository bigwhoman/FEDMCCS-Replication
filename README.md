# FEDMCCS-Replication
Replicating FEDMCCS client selection in Federated Learning
<br> 
## The paper
The paper is in the link : [FEDMCCS](https://ieeexplore.ieee.org/document/9212434)


## Running

At first run the `python3 generate_clients.py` then run the `runner.sh` with bash or sh.

If you get status code 137 it's likely caused by OOM killer. See `/var/log/kern.log`.

## Server
The server

## Test results
### Round Duration
![Round Duration](https://github.com/bigwhoman/FEDMCCS-Replication/assets/79264715/6093db09-1974-4312-a492-7d6b28b5e945)


### Convergence
![FEDMCCS and Random](https://github.com/bigwhoman/FEDMCCS-Replication/assets/79264715/f581f986-d3f5-4365-94ab-30d2425c9b02)
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
