# LLaMA-350M, TSR-AdamW, 4 L40S, 1 Node
torchrun --standalone --nproc_per_node 4 tsr_pretrain.py \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --tsr_scale 0.75 \
    --rank 384 \
    --update_proj_gap 100 \
    --proj_type full \
    --tsr_scale_emb 0.75 \
    --rank_emb 128 \
    --update_proj_gap_emb 2 \
    --proj_type_emb full \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 90000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer TSR_adamw

