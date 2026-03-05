import argparse
import os 
from llava.model.builder import load_and_merge_pretrained_model, load_and_dc_merge_pretrained_model
from llava.mm_utils import get_model_name_from_path
import shutil

def merge_lora(args):
    os.makedirs(args.save_model_path, exist_ok=True)

    # merged model name1
    exam_datasets = ['LLaVA_7B_lora_r16_IconQA',
                     'LLaVA_7B_lora_r16_flickr30k',
                     'LLaVA_7B_lora_r16_REC',
                     'LLaVA_7B_lora_r16_ScienceQA',
                     'LLaVA_7B_lora_r16_OCRVQA',
                     'LLaVA_7B_lora_r16_ImageNet',
                     'LLaVA_7B_lora_r16_VizWiz',
                     'LLaVA_7B_lora_r16_VQAv2']
    model_path = args.model_path

    # copy adapter config for merged model
    src = os.path.join(model_path, exam_datasets[0], "adapter_config.json")
    dst = os.path.join(args.save_model_path, "adapter_config.json")
    shutil.copy(src, dst)
    
    model_path = [os.path.join(model_path, exam_dataset) for exam_dataset in exam_datasets]
    model_name = get_model_name_from_path(model_path[0])
    
    # RobustMerge
    # load_and_merge_pretrained_model(model_path, args.model_base, model_name, args.save_model_path, device_map='cpu')
    
    # DC-Merge
    load_and_dc_merge_pretrained_model(model_path, args.model_base, model_name, args.save_model_path, 
                                      device_map='cpu',
                                      smoothing_strategy=args.smoothing,
                                      rho=args.rho)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--smoothing", type=str, choices=['avg', 'linear'], default='avg')
    parser.add_argument("--rho", type=float, default=5.0)

    args = parser.parse_args()

    merge_lora(args)
