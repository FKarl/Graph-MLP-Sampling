# How to run a script on the unicluster.
This is a quick tutorial on how to run Graph-MLP on the [BwUniCluster2.0](https://www.scc.kit.edu/en/services/bwUniCluster_2.0.php) service.

## Python version
BwUniCluster uses Python 3.6 by default. Pytorch requires at least Python 3.7.
Load Python 3.8 by running
```shell
module load devel/python/3.8.6_gnu_10.2
```
> IMPORTANT: The python version is not persisted! Every time you reconnect you need to run the command above BEFORE activating the virtual environment!

## Setup
First, create and activate a virtual environment using
```shell
python -m venv ./venv
source venv/bin/activate
```
Next, install PyTorch and PyTorch-Geometric (PyG).

Install the correct **PyTorch** version for your system from [the official website](https://pytorch.org/get-started/locally/).
Then install the corresponding **PyG** version (correct PyTorch version and same CUDA version) from [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

After this is done install the remaining requirements from the `requirements.txt` by running:
```shell
pip install -r requirements.txt
```

### WandB
Our implementation uses [Weights & Biases](https://wandb.ai/site) to easily track your runs.
To change the entity (personal or team account) and the project, change the variables at the top of [train.py](../train.py).
To disable logging in a run, add the `--no-wandb` flag.

To track runs you have to log in to WandB on your device. To do this activate the virtual environment and run 
```
wandb login
```


## Unicluster
Create a script or use an existing one, and define your SBATCH variables at the very top (see also [../run-scriptsrun.sh](../run-scripts/run.sh)).
See the [documentation](https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm#Job_Submission_:_sbatch) for more information.
To queue a job, simply run
```shell
sbatch run.sh
```
It is required that you define the queue you want to use (this is done in the `run.sh` script directly).
For all possible queues and their minimum, default and maximum resources, see [here](https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues#sbatch_-p_queue).
If you do not comply with the resource limitations, the job will not be accepted!
