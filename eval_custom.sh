MODEL_PATH=$1
CONFIG=/home/coder/generative-ai-research-babylm/conf/config.yaml
MODEL_BASENAME=$(basename $MODEL_PATH)

#Evaluate Blimp and Blimp supplement
python -m lm_eval --model roberta-custom \
    --model_args config_path="$CONFIG" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --output_path results/${MODEL_BASENAME}/blimp/${MODEL_BASENAME}/blimp_results.json \
    --log_samples

#Evaluate ewok
python -m lm_eval --model roberta-custom \
    --model_args config_path="$CONFIG" \
    --tasks ewok_filtered \
    --device cuda:0 \
    --batch_size 1 \
    --output_path results/${MODEL_BASENAME}/ewok/${MODEL_BASENAME}/ewok_results.json \
    --log_samples