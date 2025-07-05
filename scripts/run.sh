#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=defq
#SBATCH --cpus-per-gpu=24
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=100G
#SBATCH --time=4:00:00
#SBATCH --out logs/%j.txt
#SBATCH --container-image=/home/maximilian.schall/maxscha+code+latest.sqsh
#SBATCH --container-workdir=/workspace
#SBATCH --container-mounts=/home/maximilian.schall/grammar-aware-pre-training:/workspace,/dev/infiniband:/dev/infiniband,/home/maximilian.schall/structure-induced-hallucination/models/trained_models:/trained_models
#SBATCH --container-writable 
#SBATCH --export=ALL 

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

#srun --nodes=1 --ntasks=1 --gpus=1 --time=01:00:00 --partition=defq --container-image=/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh --container-workdir=/workspaces --container-mounts=/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code,/mnt/vast-gorilla:/workspaces/vast-gorilla --export=ALL --pty bash

# Could also just replace the next two lines with python3
#torchrun \
#    --nproc_per_node=$GPU_COUNT --nnodes=1 --master_port=$(shuf -i 29500-65535 -n 1) \
#    train.py --config_path cfgs/qwen3_instruct_1.7b.yml --num_devices=$GPU_COUNT --val_before_training False \
#    --eval_micro_batch_size=1 \
#    --micro_batch_size=1  --distributed_strategy=fsdp --precision=bf16-true --grad_clip=-1 \
#    --allowed_id_loss_scale=0 --loss_type structured_head -n qwen3_1.7b_instruct_2_weight_decay_baseline \
#    --use_peft False --final_hf_model_path /trained_models/qwen3_1.7b_instruct_2_weight_decay_baseline \
#    --weight_decay=0.1 --learning_rate=5e-5  \
#    --batch_size=16