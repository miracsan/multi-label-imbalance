import csv
import os
import argparse
from constructModel import get_num_pool_ops, load_model
from utils import get_configs_from_dataset
from dataloaders import construct_dataloaders
from makePredictions import make_pred_multilabel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        help='softmax_response, mc_dropout, extra_class, extra_head, entropy',
        default='softmax_response'
    )
    parser.add_argument(
        '--save-vars',
        type = int,
        default = 0
    )
    parser.add_argument(
        '--alias',
        help='under what folder name the models will be saved',
        default='baseline_segmentation'
    )
    parser.add_argument(
        '--dataset',
        help='nih',
        default = 'nih'
    )
    parser.add_argument(
        '--checkpoint',
        help='checkpoint, best_checkpoint',
        default = 'best_checkpoint'
    )

    
    options = parser.parse_args()
    
    
    
    model_path = os.path.join('results', options.alias, options.checkpoint)
    model, _, _, _ = load_model(model_path)
    
    _, batch_size, _ = get_configs_from_dataset(options.dataset)
    num_pool_ops = get_num_pool_ops(model.architecture)
    dataloaders = construct_dataloaders(options.dataset, batch_size, num_pool_ops, output_filenames=True)
    
    _, auc = make_pred_multilabel(model, dataloaders, options.method, options.alias, save_as_csv=True)
    