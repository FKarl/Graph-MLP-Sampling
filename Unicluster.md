# How to run a script on the unicluster.

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

Create a script or use an existing one, and define your SBATCH variables at the very top.
See the [documentation](https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm#Job_Submission_:_sbatch) for more information.
To queue a job, simply run
```shell
sbatch -p <queue> run.sh
```
It is required that you define the queue you want to use (this can also be done in the script directly).
For all possible queues and their minimum, default and maximum resources, see [here](https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues#sbatch_-p_queue).
If you do not comply with the resource limitations, the job will not be accepted!
