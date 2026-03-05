CUDA_VISIBLE_DEVICES=0 python merge_lora_weights.py --model-path checkpoints \
    --model-base llava-v1.5-7b \
    --save-model-path checkpoints/LLaVA_7B_lora_r16_dcmerge