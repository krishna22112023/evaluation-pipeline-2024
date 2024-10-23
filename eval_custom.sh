#!/bin/bash

# Enable error reporting and nullglob
set -e
shopt -s nullglob

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "yq is not installed. Please install it first."
    echo "You can install it using: sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && sudo chmod +x /usr/bin/yq"
    exit 1
fi

# Function to update YAML file
update_yaml() {
    local key="$1"
    local value="$2"
    local file="$3"
    
    # Update the YAML file
    yq e ".$key = \"$value\"" -i "$file"
    
    # Verify the update
    if [[ $(yq e ".$key" "$file") == "$value" ]]; then
        echo "Successfully updated $key in $file"
    else
        echo "Failed to update $key in $file"
        echo "Current value of $key:"
        yq e ".$key" "$file"
        exit 1
    fi
}

# Set paths
BASE_DIR="/home/coder/generative-ai-research-babylm"
CONFIG="$BASE_DIR/conf/config.yaml"
EXP_NAME="RoBERTa_WDML/train_10M_old"
NUM_PEER="4"

# Print initial config file contents
echo "Initial config file contents:"
cat "$CONFIG"
echo "------------------------"

# Update exp_name in config
update_yaml "general.exp_name" "$EXP_NAME" "$CONFIG"

# Find model files
MODEL_DIR="$BASE_DIR/models/$EXP_NAME/num_peer_$NUM_PEER"
MODEL_FILES=("$MODEL_DIR"/*.pt)

# Find config files
CONFIG_DIR="$MODEL_DIR/arch_search_results"
CONFIG_FILES=("$CONFIG_DIR"/*.json)

# Print all found files for verification
echo "Found model files:"
printf '%s\n' "${MODEL_FILES[@]}"
echo "Total model files: ${#MODEL_FILES[@]}"

echo "Found config files:"
printf '%s\n' "${CONFIG_FILES[@]}"
echo "Total config files: ${#CONFIG_FILES[@]}"

# Check if we have any files
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "Error: No .pt files found in $MODEL_DIR"
    exit 1
fi

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No .json files found in $CONFIG_DIR"
    exit 1
fi

# Check if we have matching numbers of model and config files
if [ ${#MODEL_FILES[@]} -ne ${#CONFIG_FILES[@]} ]; then
    echo "Warning: Number of model files (${#MODEL_FILES[@]}) doesn't match number of config files (${#CONFIG_FILES[@]})."
    echo "Proceeding with evaluation, but some models or configs might be skipped."
fi

# Loop through model and config files
for i in "${!MODEL_FILES[@]}"; do
    MODEL_PATH="${MODEL_FILES[$i]}"
    CONFIG_PATH="${CONFIG_FILES[$i]}"
    
    # Check if CONFIG_PATH exists (in case of mismatched file counts)
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Warning: No matching config file for $MODEL_PATH. Skipping this model."
        continue
    fi
    
    # Create relative paths
    REL_MODEL_PATH="models/$EXP_NAME/num_peer_$NUM_PEER/$(basename "$MODEL_PATH")"
    REL_CONFIG_PATH="models/$EXP_NAME/num_peer_$NUM_PEER/arch_search_results/$(basename "$CONFIG_PATH")"
    
    # Update config with relative model and config paths
    update_yaml "eval.model_name_or_path" "$REL_MODEL_PATH" "$CONFIG"
    update_yaml "eval.model_config_path" "$REL_CONFIG_PATH" "$CONFIG"
    
    # Print paths for verification
    echo "Processing model: $REL_MODEL_PATH"
    echo "With config: $REL_CONFIG_PATH"
    
    # Extract model basename for output directories
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    
    # Evaluate ewok
    python -m lm_eval --model roberta-custom \
    --model_args config_path="$CONFIG" \
    --tasks ewok_filtered \
    --device cuda:0 \
    --batch_size 1 \
    --output_path "$BASE_DIR/results/${MODEL_BASENAME}/ewok/${MODEL_BASENAME}/ewok_results.json" \
    --log_samples

    # Evaluate Blimp and Blimp supplement
    python -m lm_eval --model roberta-custom \
    --model_args config_path="$CONFIG" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --output_path "$BASE_DIR/results/${MODEL_BASENAME}/blimp/${MODEL_BASENAME}/blimp_results.json" \
    --log_samples
    
    echo "Evaluation complete for $MODEL_BASENAME"
    echo "----------------------------------------"
done

echo "All evaluations completed."