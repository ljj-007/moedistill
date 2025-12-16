#!/bin/bash
# filepath: /data01/jun/my-moe/run_expert_distill.sh

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
# export PYTHONPATH=/data01/jun/my-moe:$PYTHONPATH

# Set basic paths
BASE_DIR="./"
OUTPUT_DIR="$BASE_DIR/outputs/expert_distill_qwen3_4b_0.6b_$(date +%Y%m%d_%H%M%S)_new_mc3"
TEACHER_MODEL="./checkpoints/qwen3-4B"  # Please replace with actual teacher model path
STUDENT_MODEL="./checkpoints/qwen3-0.6B"  # Please replace with actual student model path
TEACHER_LORA="./outputs/sft_qwen3-4b_5epoch/checkpoint-14295/adapter_model"    # Please replace with actual teacher LoRA path
DATASET_DIR="./data/dolly"   # Please replace with actual data path

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting expert distillation training..."
echo "Output directory: $OUTPUT_DIR"
echo "Teacher model: $TEACHER_MODEL"
echo "Student model: $STUDENT_MODEL"
echo "Dataset: $DATASET_DIR"

# Build parameter array
ARGS=(
    --teacher_model_path "$TEACHER_MODEL"
    --student_model_path "$STUDENT_MODEL"
    --dataset_dir "$DATASET_DIR"
    --output_dir "$OUTPUT_DIR"
    --overwrite_output_dir
    --do_train
    --save_strategy steps
    --save_steps 1000
    --logging_steps 50
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 1
    --num_train_epochs 5
    --learning_rate 1e-4
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --max_seq_length 512
    --load_in_kbits 16
    --torch_dtype bfloat16
    --bf16
    --dataloader_num_workers 4
    --gradient_checkpointing
    --report_to none
    --seed 42
    --enable_student_lora
    --lora_rank 8
    --lora_alpha 32
    --lora_dropout 0.1
    --lora_nums 4
    --target_modules "gate_proj,down_proj,up_proj"
    --model_type "auto"
    --enable_expert_distillation
    --loss_type "forward_kl"
    --temperature 1.0
    --alpha 0.7
    --lam 0.9
    --expert_distill_weight 0.5
    --mc_sampling_steps 3
    --expert_alpha 0.5
    --lambda_coverage 0.5
    --initial_temperature 1.0
    --beta_entropy 0.1
    --enable_method_a
    --enable_method_b
)


if [ -n "$TEACHER_LORA" ] && [ "$TEACHER_LORA" != "" ]; then
    ARGS+=(--teacher_lora_path "$TEACHER_LORA")
    echo "Teacher LoRA: $TEACHER_LORA"
else
    echo "Teacher LoRA path not set"
fi

# Execute training
python expert_distill.py "${ARGS[@]}"

# python expert_distill.py \
#     --teacher_model_path $TEACHER_MODEL \
#     --student_model_path $STUDENT_MODEL \
#     --teacher_lora_path $TEACHER_LORA \
#     --dataset_dir $DATASET_DIR \
#     --output_dir $OUTPUT_DIR \
#     --overwrite_output_dir \
#     --do_train \
#     --save_strategy steps \
#     --save_steps 1000 \
#     --logging_steps 50 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-4 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0.01 \
#     --max_seq_length 1024 \
#     --load_in_kbits 16 \
#     --torch_dtype bfloat16 \
#     --bf16 \
#     --dataloader_num_workers 4 \
#     --gradient_checkpointing \
#     --report_to none \
#     --seed 42 \
#     --enable_student_lora \
#     --lora_rank 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --lora_nums 4 \
#     --target_modules "gate_proj,down_proj,up_proj" \
#     --enable_expert_distillation \
#     --loss_type "forward_kl" \
#     --temperature 1.0 \
#     --alpha 0.7 \
#     --lam 0.9 \
#     --expert_distill_weight 0.5 \
#     --mc_sampling_steps 3 \
#     --expert_alpha 0.5 \
#     --lambda_coverage 0.5 \
#     --initial_temperature 1.0 \
#     --beta_entropy 0.1 \
#     --enable_method_a \
#     --enable_method_b


echo "Training complete!"
echo "Model saved in: $OUTPUT_DIR"

# Check for completed file
if [ -f "$OUTPUT_DIR/completed" ]; then
    echo "Training completed successfully!"
else
    echo "Warning: Completed file not found, training may not have finished correctly"
fi

# Show output directory contents
echo "Output directory contents:"
ls -la $OUTPUT_DIR/