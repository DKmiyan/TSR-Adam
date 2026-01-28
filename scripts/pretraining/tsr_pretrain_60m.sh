# LLaMA-60M, TSR-Adam, 1 L40S, 1 Node
torchrun --standalone --nproc_per_node 1 tsr_pretrain.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --tsr_scale 0.5 \
    --rank 256 \
    --update_proj_gap 100 \
    --proj_type full \
    --tsr_scale_emb 0.5 \
    --rank_emb 64 \
    --update_proj_gap_emb 2 \
    --proj_type_emb full \
    --batch_size 256 \
    --total_batch_size 1024 \
    --num_training_steps 20000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer tsr_adamw
