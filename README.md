

### Step 1 - SSH into the cluster

```
$ ssh -Y <computing_id>@login.hpc.virginia.edu
```

### Step 2 - Clone your repo onto the cluster

```
$ cd /scratch/<computing_id>    # use scratch, not home (more space)
$ git clone <your_project>.git
$ cd <your_project>
```
But if you store your *.py* files under /scratch/<computing_id>, you are at the risk of the removal of the data & scripts after 90 days.

### Step 3 - Set up conda environment

```
$ module load miniforge (!anaconda)
$ conda create -n cmrrecon python=3.10 -y
$ conda activate cmrrecon
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
$ pip install h5py sigpy tensorboard
```

You may also dump this environment to a file so have a backup for reproducibility:
```
$ conda env export > environment.yml
```

**Update**

> [!tip] 
> The UVA HPC system (Rivanna) officially retired and removed Anaconda on October 15, 2024. They transitioned entirely to `miniforge`. The *anaconda* module is permanently gone.


### Step 4 - Transfer your data

From your **local machine**
```
$ rsync -avz /local/path/to/data/ <computing_id>@login.hpc.virginia.edu:/scartch/<computing_id>/data/
```
Or if data is already on the lab's Rivanna storage, just point to that path.

### Step 5 - Write a SLURM job script

Create ==train_job.slurm== in the project root:
```
#!/bin/bash
#SBATCH --job-name=cmrrecon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --account=<your_allocation>
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate cmrrecon

cd /home/<computing_id>/CMRxRecon/src

python train.py \
	--trn_dir /scratch/<computing_id>/data/SingleCoil/train \
	--val_dir /scrathc/<computing_id>/data/SingleCoil/Var \
	--use_gpu \
	--cascade_depth 2 \
	--num_epochs 40 \
	--batch_size 1 \
	--num_filters 32 \
	--net_depth 4 \
	--learning_rate 1e-4 \
	--scheduler cosine \
	--kspace_weight 0.1
```

Key SLURM options to know:

| Flag              | Meanging                                              |
| :---------------- | ----------------------------------------------------- |
| --partition=gpu   | GPU queue (use bii-gpu if you have BII allocation)    |
| --gres=gpu:a100:1 | Request 1 A100 (also v100, rtx2800)                   |
| --time=24:00:00   | max wall time - set generously                        |
| --account=        | Your PI's allocation (check with allocations command) |



> [!tip]
> The network only run with batch size being set to 1 with A100 GPU

### Step 6 - Submit and monitor

```
$ mkdir -p logs
$ sbatch train_job.slurm     # submit
$ squeue -u <computing_id>   # check status
$ tail -f logs/<job_id>.out  # watch live output
$ scancel <job_id>           # cancel if needed
```

### Step 7 - Monitor training with TensorBoard

Start the server and leave the terminal window running:

```
$ module load miniforge && conda activate cmrrecon
$ tensorboard --logdir /home/<computing_id>/CMRxRecon/experiments --port 6006
```

From your **local machine**, set up an SSH tunnel:

```
$ ssh -N -f -L 6006:localhost:6006 <computing_id>@login.hpc.virginia.edu
```

But if you are using `wsl2`:

```
ssh -J <computing_id>@login.hpc.virginia.edu #portnumber:localhost:#portnumber hostname@<node>
```

Open http://localhost:10086 in your local browser

> [!tip]
> in case you will conflict your port number with others, try to use one larger, such as 10086


## Q&A


**Q: How to check your allocation?**
```
$ allocations
```

**Q: How to check available GPU nodes?** 
```
$ sinfo -p gpu
```

**Q: How to interactively debug before submitting?**
```
$ ijob -A <allocation> -p gpu --gres=gpu:1 -c 4 --mem=16G
```

**Q: Why is wall-clock time, and why do you set it to 24hrs not 7 days?**

**A:** In a slurm script (`.slurm`), `--time=24:00:00` is the hard time limit (wall-clock time) you are requesting from the cluster scheduler for your job.

The format is `hours:minutes:seconds`. You are asking the cluster for exactly 24 hours of uninterrupted computing time.

Here is the unvarnished reality of how this specifically affects your deep learning job:

#### The Executioner

This is not a polit estimate; it is an absolute kill switch. If your MRI reconstruction model is still training at 24 hours and 1 second, Slurm will ruthlessly terminate the process, regardless of whether you are in the middle of a critical epoch. Your error log will simply say `CNACELLED due to TIME_LIMIT`.

#### The Checkpoint Mandate

Because training complex multi-coil CMRxRecon networks on an A100 GPU can take days, you will likely hit this 24-hour wall. You *must* write code in your PyTorch training loop to save model checkpoints (e.g. `torch.save(model.state_dict(), 'checkpoint.pth')`) every few epochs. When Slurm inevitably kills your job, you just submit a new `.slurm` script that loads that `.pth` file and resumes training exactly  where you left off. If you don't checkpoint, you will lose 24 hours of work instatly.

#### The Queue Trade-off

You might wonder why you don't just set `--time=7-00:00:00` (7 days) to avoid getting killed. HPC schedulers penalize greed. If you ask for 24 hours, the scheduler can easily slot you into a brief opening, and you might start tonight. If you ask for 7 days, you might sit in the `PENDING` queue for three weeks waiting for a massive block of time to become available. 


**Q: How to resume a killed job?**

**A:** To successfully resume a killed job, you must handle the checkpoint logic directly inside your PyTorch code. You must save the complete training state, not just the model weights, and build a mechanism to load it when the script starts.

Here is the exact implementation.

#### Update `train.py` to Save Checkpoints

You must save the model weights, the optimizer state (crucial so algorithms like Adam do not lose their momentum matrices), and the current epoch number.

Add this to the very end of you epoch loop:

```
Python

# Inside your training loop, at the end of each epoch:
checkpoint = {
	'epoch': epoch,
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict()
}

# Overwrite the same file to save storage space
torch.save(checkpoint, '/scratch/wrr6ps/CMRxRecon/experiments/checkpoint_latest.pth')
```

#### Update `train.py` to Load Checkpoints

Before your `for epoch in range(num_epochs):` loop begins, add logic to check if that checkpoint file exists. If it does, load the sates into your model and optimizer, and update the starting epoch

```
import os

# After initializing model and optimizer, but BEFORE the training loop:
checkpoint_path = '/scratch/wrr6ps/CMRxRecon/experiments/checkpoint_latest.pth'
start_epoch = 0

if os.path.exists(checkpoint_path):
	print(f"Resuming training from {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path)
	
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch'] + 1
	
else:
	print("No checkpoint found. Starting training from scratch.")
	

# Update your loop definition to use start_epoch:
for epoch in ragne(start_epoch, num_epochs):
	# ... your existing training code ...
```

#### The Slurm Workflow

You do not need to change the `#SBATCH` parameters in your `.slurm` file. Because you hardcoded checkpoint path, the workflow is completely automated:

-  You run `sbatch train.slurm`
-  The script sees no checkpoint and starts at Epoch 0.
-  It overwrites `checkpoint_latest.pth` at the end of every epoch.
-  At exactly 24:00:00, Slurm kills the job.
-  You log in and run `sbatch train.slurm` again.
-  The script finds `checkpoint_latest.pth`, loads the exact model weights and optimizer momentum, and resumes training on the exact epoch it died on.




---
### Notes

- Scratch is purged after 90 days - move results to /project storage
- Detailed training information, can be found in [[Dummy Questions for Deep Learning]]
