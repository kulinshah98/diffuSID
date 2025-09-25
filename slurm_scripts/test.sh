# make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval exp_id=dd_test2 train=false\
#  ckpt_path=/scratch/01318/kulin/diffusid/intern2-runs/2025-09-18/05-09-37/checkpoints/recallk10-targetp100-seqlen100.0.0949.ckpt\
#   logger=csv"
# make train-gpu ARGS="experiment=discrete_diffusion_train_rope\
#  train=false exp_id=test3 model.num_hierarchies=4 sid_data_path="./data/amazon/beauty/sids/flan-t5-xxl_rkmeans_3_256_seed43.pt"\
#  ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dd_beauty_test2/checkpoints/recallk10-targetp100-seqlen100.0.1024.ckpt"

make train-gpu ARGS="experiment=dd_train_correct_dense_retrieval exp_id=dd_dense_mean_linear4 ckpt_path=/scratch/01318/kulin/diffusid/logs/train/runs/dd_dense_mean_linear3/recallk10-targetp100-seqlen100.0.0582.ckpt"
