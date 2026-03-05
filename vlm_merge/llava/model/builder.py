#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os, sys
import warnings
import shutil
from collections import defaultdict, OrderedDict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x


def get_ft_parameters_delta(sd):
    layer2lora_parameters = defaultdict(lambda: dict())
    for key, val in sd.items():
        if 'lora_A.' in key:
            key = key.replace('base_model.model.', '') if key.startswith('base_model.model.') else key
            base_name = key.replace('lora_A.', '')
            layer2lora_parameters[base_name]['A'] = val
        elif 'lora_B.' in key:
            key = key.replace('base_model.model.', '') if key.startswith('base_model.model.') else key
            base_name = key.replace('lora_B.', '')
            layer2lora_parameters[base_name]['B'] = val

    task_parameters = {}
    for name, key2val in layer2lora_parameters.items():
        # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
        task_parameters[name] = (key2val['B'] @ key2val['A'])
    return OrderedDict(sorted(task_parameters.items()))


def aggregate_deltas(delta_dicts):
    agg = defaultdict(list)
    for key in delta_dicts[0]:
        for i in range(len(delta_dicts)):
            agg[key].append(delta_dicts[i][key])

    return OrderedDict(sorted(agg.items()))


def generate_linear_distribution(num_classes, ratio):
    s = torch.linspace(ratio, 1.0, num_classes)
    s = s / s.sum()
    return s


def dc_merge(deltas_dict, smoothing_strategy='avg', rho=5.0):
    dc_dict = {}


    for k, vecs in tqdm(deltas_dict.items(), desc='DC-Merge Processing...'):
        N = len(vecs)
        m, p = vecs[0].shape
        low_rank_per_task = 16
        smoothed_vecs = []
        
        for i in range(N):
            u, s, v = torch.svd_lowrank(vecs[i].to(torch.float32).cuda(), q=low_rank_per_task)
            v = v.T
            if i == 0:
                n = N * low_rank_per_task
                sum_u = torch.zeros((m, n), dtype=torch.float32, device='cuda')
                sum_v = torch.zeros((n, p), dtype=torch.float32, device='cuda')

            sum_u[:, i * low_rank_per_task : (i + 1) * low_rank_per_task] = u
            sum_v[i * low_rank_per_task : (i + 1) * low_rank_per_task, :] = v
            
            orig_energy = s[:low_rank_per_task].clone()
            if smoothing_strategy == 'linear':
                smoothed_ratio = min(rho, orig_energy[0] / orig_energy[-1])
                smoothed_energy_dist = generate_linear_distribution(low_rank_per_task, smoothed_ratio)
            elif smoothing_strategy == 'avg':
                smoothed_energy_dist = torch.ones_like(orig_energy) / len(orig_energy)
            else:
                raise ValueError("Invalid smoothing strategy")
            
            smoothed_energy = orig_energy.sum() * smoothed_energy_dist
            smoothed_vecs.append(u[:, :low_rank_per_task] @ torch.diag(smoothed_energy) @ v[:low_rank_per_task, :])
           
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        cover_space_u = (u_u @ v_u)[:, :N * low_rank_per_task]
        cover_space_vT = (u_v @ v_v)[:N * low_rank_per_task, :]

        Ms = [torch.linalg.multi_dot((cover_space_u.T, smoothed_vecs[i], cover_space_vT.T, )) for i in range(N)]
        filtered_Ms = keep_topk_percent(Ms, 1e-3)
        agg_M = ties_small(filtered_Ms)
        mask_M = torch.zeros_like(agg_M)
        d_per_task = mask_M.shape[0] // N
        for i in range(N):
            mask_M[i * d_per_task : (i+1) * d_per_task, i * d_per_task : (i+1) * d_per_task] = 1
        
        dc_dict[k] = torch.linalg.multi_dot((cover_space_u, agg_M * mask_M , cover_space_vT, ))

    return OrderedDict(sorted(dc_dict.items()))


def keep_topk_percent(tensor_list, percent=0.1):
    new_list = []
    for i, t in enumerate(tensor_list):
        flat_abs = t.abs().view(-1)
        k = max(1, int(percent * flat_abs.numel()))
        threshold = torch.topk(flat_abs, k).values.min()
        mask = t.abs() >= threshold
        new_t = t * mask
        new_list.append(new_t)
    
    return new_list


def ties_small(mat_list):
    stacked = torch.stack(mat_list, dim=0)
    summed = stacked.sum(dim=0)
    summed_sign = torch.sign(summed)
    elem_sign = torch.sign(stacked)
    mask = (elem_sign == summed_sign.unsqueeze(0))
    count = mask.sum(dim=0)
    selected = stacked * mask
    res = selected.sum(dim=0) / count.clamp(min=1)
    res = torch.where(summed == 0, torch.zeros_like(res), res)

    return res


def load_merged_model(model_path, model_base, model_name, 
                            load_8bit=False, 
                            load_4bit=False, 
                            device_map="auto", 
                            device="cuda", 
                            use_flash_attn=False, 
                            **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            lora_cfg_pretrained.use_cache = True                # Since we do not train, we can enable the cache
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            print(model_path)
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            
            print('Loading LoRA weights...')
            lora_state_dict = torch.load(os.path.join(model_path, 'merged_deltas.pt'), map_location='cpu')
            pretrained_state_dict = model.state_dict()
            for k, param in lora_state_dict.items():
                pretrained_state_dict[k].add_(2.0 * param.to(torch.float16).to(pretrained_state_dict[k].device))
            
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            lora_cfg_pretrained.use_cache = True                # Since we do not train, we can enable the cache
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            print(model_path)
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_and_merge_pretrained_model(model_paths, model_base, model_name, save_model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            
            model_path = model_paths[0] # loading mm_prejector, just for convenience 
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            
            print('Loading additional LLaVA weights...')        
            
            print('Start saving merged model weights..')
            coef, mask_ratio, att_ratio = 1/len(model_paths), 0.2, 0.2
            # 1. merge non-lora weights 
            merged_non_lora_trainables = {}

            for i, sub_model_path in enumerate(model_paths):
                assert os.path.exists(os.path.join(sub_model_path, 'non_lora_trainables.bin')), 'must load from local dir'
                non_lora_trainables = torch.load(os.path.join(sub_model_path, 'non_lora_trainables.bin'), map_location='cpu')
                
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                for name, param in non_lora_trainables.items():
                    if i==0:
                        merged_non_lora_trainables[name] = coef * param
                    else:
                        merged_non_lora_trainables[name] = merged_non_lora_trainables[name] + coef * param
            torch.save(merged_non_lora_trainables, os.path.join(save_model_path,'non_lora_trainables.bin'))
            
            # 2. load, merge and save lora weights (DIR-Merging): prune&scaling and normalization
            fuse_weight = (torch.ones(len(model_paths))*2.0)[:,None,None]
            concat_model_weights, merged_model_weights = {}, {}
            for i, sub_model_path in enumerate(model_paths):
                lora_parameters = torch.load(os.path.join(sub_model_path, 'adapter_model.bin'), map_location='cpu')
                for name, param in lora_parameters.items():
                    if i == 0:
                        concat_model_weights[name] = param[None,:]
                    else:
                        concat_model_weights[name] = torch.concat((concat_model_weights[name],param[None,:]),dim=0)

            for name, concat_model_weight in concat_model_weights.items():
                T, d1, d2 = concat_model_weight.shape
                concat_model_weight = concat_model_weight.reshape(-1, d1*d2)                

                kth_values, _ = concat_model_weight.abs().kthvalue(int(d1 * d2* (1-mask_ratio)), dim=1, keepdim=True)
                
                masks = (concat_model_weight.abs() >= kth_values).reshape(T, d1, d2)
                
                concat_model_weight = concat_model_weight.reshape(T, d1, d2)
                trimed_model_weights = masks * concat_model_weight

                assert 'lora_A' or 'lora_B' in name, 'not lora components, not implemented yet.'
                if 'lora_A' in name:
                    s_vector = torch.sum(abs(concat_model_weight),dim=-1)/ torch.sum(abs(masks * concat_model_weight), dim=-1)
                    scale = clamp(s_vector, 1-att_ratio, 0)
                    
                    merged_model_weights[name]=torch.sum(fuse_weight * trimed_model_weights, dim=0)
                else: # 'lora_B' in name
                    merged_model_weights[name]=torch.sum(fuse_weight * scale.unsqueeze(1) * trimed_model_weights, dim=0)/torch.sum(scale.unsqueeze(1), dim=0)
                    
            model.config.save_pretrained(save_model_path)
            torch.save(merged_model_weights, os.path.join(save_model_path,'adapter_model.bin'))
            print('Finish saving merged model weights.')

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return 


def load_and_dc_merge_pretrained_model(model_paths, model_base, model_name, save_model_path, 
                                      load_8bit=False, 
                                      load_4bit=False, 
                                      device_map="auto", 
                                      device="cuda",
                                      smoothing_strategy='avg',
                                      rho=5.0,
                                      use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            
            model_path = model_paths[0] # loading mm_prejector, just for convenience 
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            
            print('Loading additional LLaVA weights...')        
            
            print('Start saving merged model weights..')
            coef, mask_ratio, att_ratio = 1/len(model_paths), 0.2, 0.2
            # 1. merge non-lora weights 
            merged_non_lora_trainables = {}

            for i, sub_model_path in enumerate(model_paths):
                assert os.path.exists(os.path.join(sub_model_path, 'non_lora_trainables.bin')), 'must load from local dir'
                non_lora_trainables = torch.load(os.path.join(sub_model_path, 'non_lora_trainables.bin'), map_location='cpu')
                
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                for name, param in non_lora_trainables.items():
                    if i==0:
                        merged_non_lora_trainables[name] = coef * param
                    else:
                        merged_non_lora_trainables[name] = merged_non_lora_trainables[name] + coef * param
            torch.save(merged_non_lora_trainables, os.path.join(save_model_path,'non_lora_trainables.bin'))
            
            # 2. load, merge and save lora weights (DC-Merge)
            delta_dicts = []
            for i, sub_model_path in enumerate(model_paths):
                lora_parameters = torch.load(os.path.join(sub_model_path, 'adapter_model.bin'), map_location='cpu')
                delta_dicts.append(get_ft_parameters_delta(lora_parameters))

            agg_deltas = aggregate_deltas(delta_dicts)
            print('Aggregated deltas for all tasks.')
            merged_deltas = dc_merge(agg_deltas, smoothing_strategy=smoothing_strategy, rho=rho)
                    
            model.config.save_pretrained(save_model_path)
            torch.save(merged_deltas, os.path.join(save_model_path, 'merged_deltas.pt'))
            print('Finish saving merged model weights.')

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return 