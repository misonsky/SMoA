export num_gpus="${NUM_GPUS:-1}"
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun --master_port "$port" --nproc_per_node=$num_gpus train.py \
    --dataset "${DATASET:-boolq}" \
    --peft_type smoa \
    --bias "${BIAS:-none}" \
    --enable_grad_ckpt \
    --model_name "${MODEL_NAME:-your model name}" \
    --batch "${BATCH_SIZE:-32}" \
    --l_num "${NUM_BLOCKS:-2}" \
    --smoa_num_blocks "${SMOA_NUM_BLOCKS:-${NUM_BLOCKS:-2}}" \
    --lr "${LR:-0.001}" \
    --r_ab "${RANK_BUDGET:-8}" \
    --lora_alpha "${LORA_ALPHA:-8}" \
    --eval_strategy "${EVAL_STRATEGY:-steps}" \
    --eval_steps "${EVAL_STEPS:-100}" \
    --logging_steps "${LOGGING_STEPS:-20}" \
    --epoch "${EPOCHS:-10}" \
    --warmup "${WARMUP_STEPS:-100}" \
    --weight_decay "${WEIGHT_DECAY:-0}" \
    --target_modules "${TARGET_MODULES:-q_proj,v_proj,k_proj,up_proj,down_proj}" \
    --ds_config "${DS_CONFIG:-ds_configs/ds_config_fp16_z0.json}" \
    --beam_size "${BEAM_SIZE:-8}" \
    --seed "${SEED:-12345}" \
    ${EXTRA_ARGS}
