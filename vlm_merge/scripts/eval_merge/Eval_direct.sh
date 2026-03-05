#!/bin/bash
set -e
GPU=1
base_dir=checkpoints

sh scripts/eval_merge/1_eval_scienceqa.sh ${base_dir}/LLaVA_7B_lora_r16_ScienceQA ${GPU}
sh scripts/eval_merge/2_eval_vizwiz_caption.sh ${base_dir}/LLaVA_7B_lora_r16_VizWiz ${GPU}
sh scripts/eval_merge/3_eval_ImageNet.sh ${base_dir}/LLaVA_7B_lora_r16_ImageNet ${GPU}
sh scripts/eval_merge/4_eval_vqav2.sh ${base_dir}/LLaVA_7B_lora_r16_VQAv2 ${GPU}
sh scripts/eval_merge/5_eval_Iconqa.sh ${base_dir}/LLaVA_7B_lora_r16_IconQA ${GPU}
sh scripts/eval_merge/6_eval_flickr30k.sh ${base_dir}/LLaVA_7B_lora_r16_flickr30k ${GPU}
sh scripts/eval_merge/7_eval_grounding.sh ${base_dir}/LLaVA_7B_lora_r16_REC ${GPU}
sh scripts/eval_merge/8_eval_ocrvqa.sh ${base_dir}/LLaVA_7B_lora_r16_OCRVQA ${GPU}