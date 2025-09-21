#!/bin/bash

#SBATCH --job-name btest

### Logging
#SBATCH --output results/out_%j.out
#SBATCH --error  results/out_%j.err

### Node info
#SBATCH --account CGAI24022
#SBATCH --nodes 1
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 17:00:00


nvidia-smi
hostname
source ~/.bashrc
conda init
conda activate goatee
cd ~/diffuSID/mycode

data_path="./data/amazon_no_cap/beauty/training_62p5"
make train-gpu ARGS="experiment=discrete_diffusion_train_rope data_loading.train_dataloader_config.dataloader.data_folder=$data_path exp_id=dd_beauty_62p5"
