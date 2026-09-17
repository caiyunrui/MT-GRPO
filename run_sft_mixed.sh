#!/bin/bash
# 解决之前遇到过的 Numba/LLVM 问题
# export NUMBA_DISABLE_JIT=1
# export NCCL_P2P_DISABLE=1

# MS-Swift 8卡启动命令
NPROC_PER_NODE=8 swift sft \
    --model_type qwen2_5_omni \
    --model /mmu-audio-ssd/yaron/LLMs/Qwen2.5-Omni-7B \
    --dataset mixed_cot_sft_train.jsonl \
    --val_dataset mixed_cot_sft_val.jsonl \
    --dataloader_num_workers 8 \
    --target_regex '^(thinker[.]model.*[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|(?!(thinker[.]audio_tower[.]proj$))thinker[.]audio_tower.*[.](k_proj|q_proj|v_proj|out_proj|fc1|fc2)|thinker[.]audio_tower[.]proj)$' \
    --train_type lora \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 200 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --output_dir /mmu-audio-ssd/yaron/SpeechLLM/swift_qwen2.5_omni_sft_mixed_train \
    --use_liger_kernel true
    # --deepspeed default-zero2

    
