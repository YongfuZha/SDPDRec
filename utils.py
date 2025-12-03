# coding: utf-8
import os
import re
import yaml
import torch
import random
import logging
import datetime
import numpy as np
import importlib
from logging import getLogger
import pickle

class NoOp(object):
    def __getattr__(self, name):
        return self.noop
     
    def noop(self, *args, **kwargs):
        return

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def zero_none_grad(model):
    for p in model.parameters():
        if p.grad is None and p.requires_grad:
            p.grad = p.data.new(p.size()).zero_()

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
    return cur

def get_model(model_name):
    if model_name == 'FREEDOM':
        from smore_model import FREEDOM
        return FREEDOM
    elif model_name == 'LightGCN_Encoder':
        from smore_model import LightGCN_Encoder
        return LightGCN_Encoder
    else:
        raise ValueError(f'Unknown model: {model_name}')

def get_trainer():
    from training import Trainer
    return Trainer

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def early_stopping(value, best, cur_step, max_step, bigger=True):
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag

def dict2str(result_dict):
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def init_logger(config):
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config.get('state') is None or config.get('state', '').lower() == 'info':
        level = logging.INFO
    elif config.get('state', '').lower() == 'debug':
        level = logging.DEBUG
    elif config.get('state', '').lower() == 'error':
        level = logging.ERROR
    elif config.get('state', '').lower() == 'warning':
        level = logging.WARNING
    elif config.get('state', '').lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers = [sh, fh]
    )

class Config(object):
    def __init__(self, model=None, dataset=None, config_dict=None, mg=False):
        if config_dict is None:
            config_dict = {}
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        
        self.final_config_dict = self._load_dataset_model_config(config_dict, mg)
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict, mg):
        file_config_dict = dict()
        
        cur_dir = os.getcwd()
        config_file = os.path.join(cur_dir, "config.yaml")
        
        hyper_parameters = []
        if os.path.isfile(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                fdata = yaml.load(f.read(), Loader=self._build_yaml_loader())
                
                if 'general' in fdata:
                    general_config = fdata['general']
                    file_config_dict.update(general_config)
                    if general_config.get('hyper_parameters'):
                        hyper_parameters.extend(general_config['hyper_parameters'])
                
                if 'datasets' in fdata and config_dict['dataset'] in fdata['datasets']:
                    dataset_config = fdata['datasets'][config_dict['dataset']]
                    file_config_dict.update(dataset_config)
                    
                if 'models' in fdata and config_dict['model'] in fdata['models']:
                    model_config = fdata['models'][config_dict['model']]
                    file_config_dict.update(model_config)
                    if model_config.get('hyper_parameters'):
                        hyper_parameters.extend(model_config['hyper_parameters'])
                
                if mg and 'mg' in fdata:
                    mg_config = fdata['mg']
                    file_config_dict.update(mg_config)
                    if mg_config.get('hyper_parameters'):
                        hyper_parameters.extend(mg_config['hyper_parameters'])
                        
        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric_str = self.final_config_dict.get('valid_metric', 'Recall@20')
        if valid_metric_str is None:
            valid_metric_str = 'Recall@20'
        valid_metric = valid_metric_str.split('@')[0].lower()
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        
        if 'hyper_parameters' not in self.final_config_dict:
            self.final_config_dict['hyper_parameters'] = []
        
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']

    def _init_device(self):
        use_gpu = self.final_config_dict.get('use_gpu', True)
        gpu_id = self.final_config_dict.get('gpu_id', 0)
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def get(self, key, default=None):
        if key in self.final_config_dict:
            return self.final_config_dict[key]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()

def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    from data_processing import RecDataset, TrainDataLoader, EvalDataLoader
    from itertools import product

    
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    
    print('Model: \t' + config['model'])
    print('Dataset: \t' + config['dataset'])
    print('Key Hyperparameters:')
    key_params = ['embedding_size', 'learning_rate', 'n_ui_layers', 'n_mm_layers', 'knn_k', 'feat_embed_dim', 'mm_image_weight', 'reg_weight', 'dropout']
    for param in key_params:
        if config[param] is not None:
            print(f'  {param}: {config[param]}')

    dataset = RecDataset(config)
    print('Dataset Statistics:')
    print(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        param_value = config[i]
        if isinstance(param_value, list):
            hyper_ls.append(param_value)
        else:
            hyper_ls.append([param_value] if param_value is not None else [None])
    
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    tsne_dataset={}
    fname = "./TSNE/clothing.pkl"
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        train_data.pretrain_setup()
        model_instance = get_model(config['model'])(config, train_data).to(config['device'])

        tsne_dataset["ori_user_emb"]=model_instance.user_embedding.weight.data.cpu().detach().numpy()
        tsne_dataset["ori_item_emb"]=model_instance.item_id_embedding.weight.data.cpu().detach().numpy()
        tsne_dataset["ori_item_image_feat"] = model_instance.image_trs(
            model_instance.image_embedding.weight).cpu().detach().numpy()
        tsne_dataset["ori_item_text_feat"] = model_instance.text_trs(
            model_instance.text_embedding.weight).cpu().detach().numpy()
        trainer = get_trainer()(config, model_instance, mg)
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        test_result = hyper_ret[best_test_idx][2]
        result_str = ', '.join([f'{k}: {v:.4f}' for k, v in test_result.items()])
        params_str = ', '.join([f'{k}={v}' for k, v in zip(config['hyper_parameters'], hyper_ret[best_test_idx][0])])
        print(f'Current Best - {params_str}, {result_str}')

        tsne_dataset["final_user_emb"] = model_instance.user_embedding.weight.data.cpu().detach().numpy()
        tsne_dataset["final_item_emb"] = model_instance.item_id_embedding.weight.data.cpu().detach().numpy()
        tsne_dataset["final_item_image_feat"] = model_instance.image_trs(model_instance.image_embedding.weight).cpu().detach().numpy()
        tsne_dataset["final_item_text_feat"] = model_instance.text_trs(model_instance.text_embedding.weight).cpu().detach().numpy()
        tsne_dataset["users_items_index"] = model_instance.users_items_index
    with open(fname,"wb") as f:
        pickle.dump(tsne_dataset,f)
    print('\n=== FINAL BEST RESULT ===')
    test_result = hyper_ret[best_test_idx][2]
    result_str = ', '.join([f'{k}: {v:.4f}' for k, v in test_result.items()])
    params_str = ', '.join([f'{k}={v}' for k, v in zip(config['hyper_parameters'], hyper_ret[best_test_idx][0])])
    print(f'Parameters: {params_str}')
    print(f'Results: {result_str}')