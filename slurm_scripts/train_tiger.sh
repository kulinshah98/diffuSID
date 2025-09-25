#!/bin/bash

#SBATCH --job-name tig625

### Logging
#SBATCH --output results/out_%j.out
#SBATCH --error  results/out_%j.err

### Node info
#SBATCH --account CGAI24022
#SBATCH --nodes 1
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 6:00:00


nvidia-smi
hostname
source ~/.bashrc
conda init
conda activate goatee
cd ~/diffuSID/mycode

# make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval exp_id=dd_dense_mean_linear4 ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dd_dense_mean_linear3/recallk10-targetp100-seqlen100.0.0582.ckpt"

make train-gpu ARGS="experiment=amazon_p5_sem_ids_train exp_id=tiger_37p5_try3"

# make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval_low_rank ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dense_lowranklinear_concat_halfway/checkpoints/recallk10-targetp100-seqlen100.0.0858.ckpt exp_id=dense_lowranklinear_concat_halfway_beam_search_cont_train"

#data_path="./data/amazon_no_cap/beauty/training_62p5"
#make train-gpu ARGS="experiment=discrete_diffusion_train_rope data_loading.train_dataloader_config.dataloader.data_folder=$data_path exp_id=dd_beauty_62p5"
