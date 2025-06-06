export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0 python distill_rao_5.py  \
--dataset sunrgbd  \
--log_dir log_sunrgbd  \
--if_wandb  \
--checkpoint_path checkpoint_99.tar \
--finetune  \
--learning_rate 0.0005  \
--batch_size 8 \
--max_epoch 100  \
--lr_decay_steps 40,80  \
--lr_decay_rates 0.1,0.1 \
# --use_imvotenet  \
--dist_url tcp://localhost:32955 \