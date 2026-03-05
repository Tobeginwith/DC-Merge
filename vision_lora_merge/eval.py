import os
import torch
import argparse
from copy import deepcopy
import logging
from datetime import datetime

import numpy as np
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
import json

from utils import *
from merging_functions import TA, TSVM, iso_cts, dc_merge, wudi_merge
from ft_handlers import get_ft_parameters_delta, aggregate_deltas, apply_merge

def get_parser():
    parser = argparse.ArgumentParser(description="Experiment Settings")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=[
            "vitB16_r16_8task",
            "vitB16_r16_12task",
            "vitB16_r16_16task",
            "vitB32_r16_8task",
            "vitB32_r16_12task",
            "vitB32_r16_16task",
            "vitL14_r16_8task",
            "vitL14_r16_12task",
            "vitL14_r16_16task"
        ],
        help="Configuration name to use.",
    )
    parser.add_argument(
        "--log_base_path",
        type=str,
        default="./logs",
        help="Base path for logging results.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["dc_merge", "tsvm", "ta", "iso_cts", "wudi"],
        help="Merging method to use.",
    )
    parser.add_argument(
        "--use_official_knots",
        action='store_true',
        help="Checkpoints used for merging. Set it when using KnOTS checkpoints.",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        choices=['avg', 'linear'],
        default='avg',
        help='Energy Strategy to use in DC-Merge (averaging by default).',
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=5.0,
        help='The hyperparameter in linear smoothing (5.0 by default).',
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=300,
        help='The iteration rounds in WUDI (300 by default).',
    )
    parser.add_argument(
        "--k_frac",
        type=float,
        default=0.8,
        help='The fraction of common space dimension in Iso-CTS (0.8 by default).',
    )

    return parser


def single_task_accuracies(base_dir="single_task", model_type="ViT-B-32"):
    model_dir = os.path.join(base_dir, model_type)
    val_path = os.path.join(model_dir, "val_acc.json")
    test_path = os.path.join(model_dir, "test_acc.json")

    with open(val_path, "r") as f:
        val_acc = json.load(f)

    with open(test_path, "r") as f:
        test_acc = json.load(f)

    return {'val': val_acc, 'test': test_acc}


def main():
    EVAL_SPLIT = 'val'
    BIGSEED = 400
    set_seed(BIGSEED)
    # Get config
    parser = get_parser()
    args = parser.parse_args()
    CONFIG_NAME = args.config
    print("Running with config: ", CONFIG_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    print("Datasets: ", dataset_names)
    dataloaders = np.array([i for i in config['data']])

    print('Creating Merge')
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    if args.use_official_knots:
        method_dir = os.path.join(args.log_base_path, args.method, 'official_knots', config['model']['base_type'])
        os.makedirs(method_dir, exist_ok=True)
        single_task_accs = single_task_accuracies(base_dir='single_task/official_knots', model_type=config['model']['base_type'])
    else:
        method_dir = os.path.join(args.log_base_path, args.method, 'our_finetuned', config['model']['base_type'])
        os.makedirs(method_dir, exist_ok=True)
        single_task_accs = single_task_accuracies(base_dir='single_task/our_finetuned', model_type=config['model']['base_type'])

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename=os.path.join(method_dir, f"{timestamp}.txt"),
    )
    with torch.no_grad():
        # initialize merging function
        models = np.array([i.cpu() for i in config['models']['bases']])
        delta_dicts = [get_ft_parameters_delta(models[i]) for i in range(len(models))]
        agg_deltas = aggregate_deltas(delta_dicts)

        if args.method == 'dc_merge':
            merged_deltas = dc_merge(agg_deltas, smoothing_strategy=args.smoothing, rho=args.rho)
        elif args.method == 'tsvm':
            merged_deltas = TSVM(agg_deltas)
        elif args.method == 'wudi':
            merged_deltas = wudi_merge(agg_deltas, iter_num=args.iter_num)
        elif args.method == 'ta':
            merged_deltas = TA(agg_deltas)
        elif args.method == 'iso_cts':
            merged_deltas = iso_cts(agg_deltas, k_frac=args.k_frac)
        else:
            raise NotImplementedError

        print('-' * 50 + 'Evaluating Merged Model on val set' + '-' * 50)
        print('\n')
        logging.info('-' * 50 + 'Evaluating Merged Model on val set' + '-' * 50)
        logging.info('\n')
        alphas = np.arange(0.1, 3.1, 0.1)
        best_alpha = alphas[0]
        best_norm_acc = 0.0

        for a in alphas:
            print(f'Using alpha = {a}')
            logging.info(f'Using alpha = {a}')
            merged_model = apply_merge(config['models']['new'], merged_deltas, scaling_coeffs=a)
            avg_accuracy = 0.
            avg_norm_accuracy = 0.
            for i, loader_dict in enumerate(dataloaders):
                loader = loader_dict['test'][EVAL_SPLIT]
                acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
                print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100) / single_task_accs[EVAL_SPLIT][dataset_names[i]] *100, 3)}")
                print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
                logging.info(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ single_task_accs[EVAL_SPLIT][dataset_names[i]] *100, 3)}")
                logging.info(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")

                avg_accuracy += acc * 100
                avg_norm_accuracy += (acc * 100) / single_task_accs[EVAL_SPLIT][dataset_names[i]] * 100
            avg_accuracy /= len(dataloaders)
            avg_norm_accuracy /= len(dataloaders)

            if avg_norm_accuracy > best_norm_acc:
                best_norm_acc = avg_norm_accuracy
                best_alpha = a
            
            print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
            print('\n\n')
            logging.info(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
            logging.info(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
            logging.info('\n\n')
        
        print('-' * 50)
        print(f'Best Alpha: {best_alpha}')
        print(f'Best Average Normalized Accuracy: {np.round(best_norm_acc, 3)}')
        logging.info('-' * 50)
        logging.info(f'Best Alpha: {best_alpha}')
        logging.info(f'Best Average Normalized Accuracy: {np.round(best_norm_acc, 3)}')
        
        print('-' * 50 + 'Evaluating Merged Model on test set' + '-' * 50)
        print('\n')
        logging.info('-' * 50 + 'Evaluating Merged Model on test set' + '-' * 50)
        logging.info('\n')
        
        merged_model = apply_merge(config['models']['new'], merged_deltas, scaling_coeffs=best_alpha)
        EVAL_SPLIT = 'test'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ single_task_accs[EVAL_SPLIT][dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            logging.info(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ single_task_accs[EVAL_SPLIT][dataset_names[i]] *100, 3)}")
            logging.info(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")

            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / single_task_accs[EVAL_SPLIT][dataset_names[i]] *100
        
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        print('\n\n')
        logging.info(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        logging.info(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        logging.info('\n\n')
    
        
    print('Finished!')

if __name__ == "__main__":
    main()

