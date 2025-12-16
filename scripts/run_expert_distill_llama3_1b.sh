#!/bin/bash
# filepath: /data3/jun/my-moe/run_expert_distill_llama3.sh
# Llama3.1-8B -> Llama3.2-1B expert distillation script

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
# export PYTHONPATH=/root/miniconda3:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set basic paths
BASE_DIR="./"
OUTPUT_DIR="./outputs/expert_distill_llama3_8b_1b_$(date +%Y%m%d_%H%M%S)_new_mc3"
TEACHER_MODEL="./checkpoints/llama-3.1-8b"  # Teacher model: Llama3.1-8B
STUDENT_MODEL="./checkpoints/llama-3.2-1b"  # Student model: Llama3.2-1B
TEACHER_LORA="./outputs/sft_llama3-8b_5epoch/checkpoint-14295/adapter_model"    # Teacher LoRA path (if any)
DATASET_DIR="./data/dolly"   # Data path

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting Llama3 expert distillation training..."
echo "Output directory: $OUTPUT_DIR"
echo "Teacher model: $TEACHER_MODEL (Llama3.1-8B)"
echo "Student model: $STUDENT_MODEL (Llama3.2-1B)"
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
    --per_device_train_batch_size 1    # Llama3.1-8B is large, reduce batch size
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 1   # Increase gradient accumulation steps to compensate for smaller batch size
    --num_train_epochs 4
    --learning_rate 5e-5               # Llama3 typically uses smaller learning rates
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --max_seq_length 512              # Llama3 supports longer sequences
    --load_in_kbits 16                  # Use 16-bit quantization to save VRAM
    --torch_dtype bfloat16
    --bf16
    --dataloader_num_workers 4
    --report_to none
    --seed 42
    --enable_student_lora
    --lora_rank 8                     # Llama3 typically uses larger rank
    --lora_alpha 32
    --lora_dropout 0.1                # Smaller dropout
    --lora_nums 4
    --target_modules "gate_proj,down_proj,up_proj"  # Consistently use these target modules
    --enable_expert_distillation
    --loss_type "forward_kl"
    --temperature 1.0                  # Higher temperature for large model distillation
    --alpha 0.8                        # More emphasis on distillation loss
    --lam 0.9
    --expert_distill_weight 0.5        # Keep expert distillation weight unchanged
    --mc_sampling_steps 3
    --expert_alpha 0.5
    --lambda_coverage 0.5
    --initial_temperature 4.0
    --beta_entropy 0.1
    --enable_method_a
    --enable_method_b
    --model_type "llama"               # Specify model type
    --max_grad_norm 1.0                # Gradient clipping
)

# Only add this parameter when TEACHER_LORA is not empty
if [ -n "$TEACHER_LORA" ] && [ "$TEACHER_LORA" != "" ]; then
    ARGS+=(--teacher_lora_path "$TEACHER_LORA")
    echo "Teacher LoRA: $TEACHER_LORA"
else
    echo "Teacher LoRA path not set"
fi

echo "Parameter list:"
printf '%s\n' "${ARGS[@]}"

# Execute training
echo "Starting training..."
python expert_distill.py "${ARGS[@]}"

echo "Training completed!"
echo "Model saved at: $OUTPUT_DIR"

# Check if there is a completed file
if [ -f "$OUTPUT_DIR/completed" ]; then
    echo "Training successfully completed!"
else
    echo "Warning: completed file not found, training may not have finished properly"
fi

# Display output directory contents
echo "Output directory contents:"
ls -la $OUTPUT_DIR/

# Display model size comparison
echo ""
echo "Model size comparison:"
echo "Teacher model (Llama3.1-8B): ~16GB"
echo "Student model (Llama3.2-1B): ~2GB"
echo "Expected compression ratio: ~8:1"
