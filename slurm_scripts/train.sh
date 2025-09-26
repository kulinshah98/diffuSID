#!/bin/bash

#SBATCH --job-name dense

### Logging
#SBATCH --output results/out_%j.out
#SBATCH --error  results/out_%j.err

### Node info
#SBATCH --account CGAI24022
#SBATCH --nodes 1
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 22:00:00


nvidia-smi
hostname
source ~/.bashrc
conda init
conda activate goatee
cd ~/diffuSID/mycode

# make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval exp_id=dd_dense_mean_linear4 ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dd_dense_mean_linear3/recallk10-targetp100-seqlen100.0.0582.ckpt"

make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval_low_rank\
 ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dense_lrlinear_cathalf_freq10_truerank32_lrp1e3/recallk10-targetp100-seqlen100.0.0909.ckpt\
 exp_id=dense_lrlinear_cathalf_freq10_truerank32_lrp1e3_cont_train"

# make train-gpu ARGS="experiment=dd_train_dense_lr_item_proj\
#   model.dense_retrieval.loss_weight=0.1\
#   exp_id=dense_item_proj_rank16_losswt0p1"

#data_path="./data/amazon_no_cap/beauty/training_62p5"
# make train-gpu ARGS="experiment=discrete_diffusion_train_rope exp_id=beauty4"
