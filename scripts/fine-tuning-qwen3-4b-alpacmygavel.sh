export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
export FLASH_ATTENTION_FORCE_DISABLE=1
export DISABLE_FLASH_ATTN=1

# Limit Intel Extension for PyTorch
export INTEL_EXTENSION_FOR_PYTORCH_DISABLE=1
export IPEX_DISABLE=1

# User-defined parameters
lr=0.0002
lora_rank=8
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.1                 
pretrained_model="./checkpoints/qwen3-4B"
tokenizer_path="./checkpoints/qwen3-4B"
dataset_dir="./data/alpacmygavel"
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=4
max_seq_length=512
output_dir="./outputs"
exp_name=sft_qwen3-4b_5epoch_alpacmygavel

lora_b_nums=4  # Developer-specific, k-means, or DBSCAN et al.



CUDA_VISIBLE_DEVICES=1 \
python finetune.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 41 \
    --bf16 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_b_nums} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --overwrite_output_dir \