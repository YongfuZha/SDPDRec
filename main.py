# coding: utf-8
import os
import argparse
from utils import quick_start


os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FREEDOM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')
    parser.add_argument('--mg', action='store_true', help='enable mg mode')
    
    parser.add_argument('--n_ui_layers', type=int, default=3, help='number of UI layers')
    parser.add_argument('--n_mm_layers', type=int, default=1, help='number of MM layers')
    parser.add_argument('--knn_k', type=int, default=10, help='KNN k for multimodal features')
    parser.add_argument('--feat_embed_dim', type=int, default=64, help='feature embedding dimension')
    parser.add_argument('--mm_image_weight', type=float, default=0.1, help='image weight in multimodal fusion')
    parser.add_argument('--reg_weight', type=float, default=1e-04, help='regularization weight')
    parser.add_argument('--dropout', type=float, default=0.8, help='dropout rate for edge pruning')
    parser.add_argument('--degree_ratio', type=float, default=1.0, help='degree ratio for edge pruning')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--embedding_size', type=int, default=64, help='embedding size')
    
    config_dict = {
        'gpu_id': 0,
    }
    
    args, _ = parser.parse_known_args()
    
    config_dict['n_ui_layers'] = args.n_ui_layers
    config_dict['n_mm_layers'] = args.n_mm_layers
    config_dict['knn_k'] = args.knn_k
    config_dict['feat_embed_dim'] = args.feat_embed_dim
    config_dict['mm_image_weight'] = args.mm_image_weight
    config_dict['reg_weight'] = args.reg_weight
    config_dict['dropout'] = args.dropout
    config_dict['degree_ratio'] = args.degree_ratio
    config_dict['learning_rate'] = args.learning_rate
    config_dict['embedding_size'] = args.embedding_size
    
    config_dict['hyper_parameters'] = []
    
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)