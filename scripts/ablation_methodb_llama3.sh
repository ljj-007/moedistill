#!/bin/bash
# Rouge-L evaluation script - Test trained models on multiple datasets

# Set default parameters
MODEL_PATH="./checkpoints/llama-3.2-1b"
LORA_PATH="./ablation_outputs/ablation_methodb_llama3/checkpoint-22870/adapter_model"
MODEL_TYPE="auto"
DATA_DIR="./data"
OUTPUT_DIR="./ablation_results/ablation_methodb_llama3"
MAX_SAMPLES="100"
DEVICE="cuda"

# Set CUDA device (force single GPU to avoid device mismatch issues)
export CUDA_VISIBLE_DEVICES=0

# Add Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --model_path <model_path> [other_options]"
            echo ""
            echo "Required parameters:"
            echo "  --model_path       Base model path"
            echo ""
            echo "Optional parameters:"
            echo "  --lora_path        LoRA weights path"
            echo "  --model_type       Model type (auto/llama/qwen), default: auto"
            echo "  --data_dir         Data directory, default: ./data"
            echo "  --output_dir       Result output directory, default: ./eval_results_<timestamp>"
            echo "  --max_samples      Maximum samples per dataset (for quick testing)"
            echo "  --device           Computing device, default: cuda"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_path ./checkpoints/llama-3.2-1b"
            echo "  $0 --model_path ./checkpoints/qwen3-0.6B --lora_path ./outputs/sft_qwen3-0.6b_3epoch/adapter_model"
            echo "  $0 --model_path ./checkpoints/llama-3.2-1b --max_samples 100  # Quick test"
            echo ""
            echo "Supported datasets: sinst, uinst, vicuna, self-inst, dolly"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help to see usage"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Must specify --model_path parameter"
    echo "Use --help to see usage"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check LoRA path (if specified)
if [ -n "$LORA_PATH" ] && [ ! -d "$LORA_PATH" ]; then
    echo "Warning: LoRA path does not exist: $LORA_PATH, will ignore LoRA weights"
    LORA_PATH=""
fi

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==================== Rouge-L Evaluation Configuration ===================="
echo "Model path:       $MODEL_PATH"
echo "LoRA path:        ${LORA_PATH:-Not specified}"
echo "Model type:       $MODEL_TYPE"
echo "Data directory:   $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Max samples:      ${MAX_SAMPLES:-Unlimited}"
echo "Computing device: $DEVICE"
echo "=========================================================="
echo ""

# Build Python command
PYTHON_CMD="python eval_rouge_l.py --model_path \"$MODEL_PATH\" --model_type \"$MODEL_TYPE\" --data_dir \"$DATA_DIR\" --output_dir \"$OUTPUT_DIR\" --device \"$DEVICE\""

if [ -n "$LORA_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --lora_path \"$LORA_PATH\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_samples $MAX_SAMPLES"
fi

echo "Starting evaluation..."
echo "Executing command: $PYTHON_CMD"
echo ""

# Install necessary dependencies (if needed)
echo "Checking and installing dependencies..."
pip install rouge-score pandas tqdm > /dev/null 2>&1

# Execute evaluation
eval $PYTHON_CMD

# Check execution result
if [ $? -eq 0 ]; then
    echo ""
    echo "==================== Evaluation Completed ===================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Main output files:"
    echo "  - summary_results.json      Summary results"
    echo "  - all_detailed_results.json All detailed results"
    echo "  - *_detailed_results.json   Detailed results for each dataset"
    echo "  - *_results.csv            CSV format results for each dataset"
    echo "=================================================="
    
    # Show output directory contents
    echo "Output directory contents:"
    ls -la "$OUTPUT_DIR"
else
    echo ""
    echo "Error occurred during evaluation, please check log information"
    exit 1
fi
