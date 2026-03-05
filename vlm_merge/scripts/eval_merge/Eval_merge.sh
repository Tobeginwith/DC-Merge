GPU=0
eval_path=checkpoints/LLaVA_7B_lora_r16_dcmerge

(
sh scripts/eval_merge/1_eval_scienceqa.sh ${eval_path} 0
sh scripts/eval_merge/2_eval_vizwiz_caption.sh ${eval_path} 0
) &

(
sh scripts/eval_merge/3_eval_ImageNet.sh ${eval_path} 1
sh scripts/eval_merge/4_eval_vqav2.sh ${eval_path} 1
) &

(
sh scripts/eval_merge/5_eval_Iconqa.sh ${eval_path} 2
sh scripts/eval_merge/6_eval_flickr30k.sh ${eval_path} 2
) &

(
sh scripts/eval_merge/7_eval_grounding.sh ${eval_path} 3
sh scripts/eval_merge/8_eval_ocrvqa.sh ${eval_path} 3
) &

wait


sh scripts/eval_merge/eval_aokvqa.sh ${eval_path} 0 &
sh scripts/eval_merge/eval_imagenetr.sh ${eval_path} 1 &
sh scripts/eval_merge/eval_screen2words.sh ${eval_path} 2 &
sh scripts/eval_merge/eval_tabmwp.sh ${eval_path} 3 &

wait
