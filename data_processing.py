# coding: utf-8
import os
import pandas as pd
import numpy as np
import torch
import random
import math
import io
from logging import getLogger
from collections import Counter
from scipy.sparse import coo_matrix
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize
from torch.nn.functional import interpolate as img_tensor_resize
from torch.nn.functional import pad as img_tensor_pad
from torch.nn.modules.utils import _quadruple
import numbers

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def flat_list_of_lists(l):
    return [item for sublist in l for item in sublist]

def mask_batch_text_tokens(inputs, tokenizer, mlm_probability=0.15, is_train=True):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape,
        dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    if not isinstance(image, (np.ndarray,)):
        raise TypeError("Input type must be a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor

def get_padding(image, max_w, max_h, pad_all=False):
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_padding, v_padding = max_w - w, max_h - h
    if pad_all:
        h_padding /= 2
        v_padding /= 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    else:
        l_pad, t_pad = 0, 0
        r_pad, b_pad = h_padding, v_padding
    if isinstance(image, torch.Tensor):
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    else:
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class ImagePad(object):
    def __init__(self, max_w, max_h, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.max_w = max_w
        self.max_h = max_h
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            paddings = _quadruple(get_padding(img, self.max_w, self.max_h))
            return img_tensor_pad(
                img, paddings,
                self.padding_mode, self.fill)
        return img_pad(
            img, get_padding(img, self.max_w, self.max_h),
            self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.format(
            self.max_w, self.fill, self.padding_mode)

def get_resize_size(image, max_size):
    if isinstance(image, torch.Tensor):
        height, width = image.shape[-2:]
    else:
        width, height = image.size

    if height >= width:
        ratio = width*1./height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height*1./width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))
    return size

class ImageResize(object):
    def __init__(self, max_size, interpolation=Image.BILINEAR):
        assert isinstance(max_size, int)
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            assert isinstance(self.interpolation, str)
            return img_tensor_resize(
                img, size=get_resize_size(img, self.max_size),
                mode=self.interpolation, align_corners=False)
        return img_resize(
            img, get_resize_size(img, self.max_size), self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.max_size, interpolate_str)

def get_imagenet_transform(min_size=600, max_size=1000):
    if min_size != 600:
        import warnings
        warnings.warn(f'Warning: min_size is not used in image transform, '
                      f'setting min_size will have no effect.')
    return transforms.Compose([
        ImageResize(max_size, Image.BILINEAR),
        ImagePad(max_size, max_size),
    ])

class ImageNorm(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 3, 1, 1)

    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i*chunk_size: (i+1)*chunk_size])
    return chunked_examples

def mk_input_group(key_grouped_examples, max_n_example_per_group=2, is_train=True,
                   example_unique_key=None):
    input_groups = []
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(examples,
                                      chunk_size=max_n_example_per_group,
                                      pad_to_divisible=is_train)
        for c in chunked_examples:
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(f"Using example_unique_key {example_unique_key} to check whether input and output ids match")
        input_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e] for e in key_grouped_examples.values()])
        output_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e[1]] for e in input_groups])
        assert set(input_question_ids) == set(output_question_ids), "You are missing some examples"
    return input_groups

def repeat_tensor_rows(raw_tensor, row_repeats):
    assert len(raw_tensor) == len(row_repeats), "Has to be the same length"
    if sum(row_repeats) == len(row_repeats):
        return raw_tensor
    else:
        indices = torch.LongTensor(
            flat_list_of_lists([[i] * r for i, r in enumerate(row_repeats)])
        ).to(raw_tensor.device)
        return raw_tensor.index_select(0, indices)

def load_decompress_img_from_lmdb_value(lmdb_value):
    io_stream = io.BytesIO(lmdb_value)
    img = Image.open(io_stream, mode="r")
    return img

class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return

        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        self.inter_num = len(self.df)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        nxt = RecDataset(self.config, new_df)
        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        nxt.inter_num = len(new_df)
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.df, self.uid_field, self.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)

class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config['device']

        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):
        pass

    def data_preprocess(self):
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

class TrainDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        self.history_items_per_u = dict()
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_items_set = set(self.all_items)
        self.all_users_set = set(self.all_uids)
        self.all_item_len = len(self.all_items)
        self.use_full_sampling = config['use_full_sampling']

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample
            else:
                self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()
        # print("self:",self.history_items_per_u)
        self.neighborhood_loss_required = config.get('use_neighborhood_loss', False)
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}
            self._get_history_users_i()
            self.user_user_dict = self._get_my_neighbors(self.config['USER_ID_FIELD'])
            self.item_item_dict = self._get_my_neighbors(self.config['ITEM_ID_FIELD'])

    def pretrain_setup(self):
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_items.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        random.shuffle(self.all_items)

    def inter_matrix(self, form='coo', value_field=None):
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        if self.use_full_sampling:
            return len(self.all_uids)
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        u_ids = cur_data[self.config['USER_ID_FIELD']]
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)
        if self.neighborhood_loss_required:
            i_ids = cur_data[self.config['ITEM_ID_FIELD']]
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(i_ids, self.config['ITEM_ID_FIELD'])
            pos_neighbors, neg_neighbors = pos_neighbors.to(self.device), neg_neighbors.to(self.device)

            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0),
                                      pos_neighbors.unsqueeze(0), neg_neighbors.unsqueeze(0)))

        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))

        return batch_tensor

    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        return batch_tensor

    def _get_full_uids_sample(self):
        user_tensor = torch.tensor(self.all_uids[self.pr: self.pr + self.step]).type(torch.LongTensor).to(self.device)
        self.pr += self.step
        return user_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            iid = self._random()
            while iid in self.history_items_per_u[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def _get_my_neighbors(self, id_str):
        ret_dict = {}
        a2b_dict = self.history_items_per_u if id_str == self.config['USER_ID_FIELD'] else self.history_users_per_i
        b2a_dict = self.history_users_per_i if id_str == self.config['USER_ID_FIELD'] else self.history_items_per_u
        for i, j in a2b_dict.items():
            k = set()
            for m in j:
                k |= b2a_dict.get(m, set()).copy()
            k.discard(i)
            ret_dict[i] = k
        return ret_dict

    def _get_neighborhood_samples(self, ids, id_str):
        a2a_dict = self.user_user_dict if id_str == self.config['USER_ID_FIELD'] else self.item_item_dict
        all_set = self.all_users_set if id_str == self.config['USER_ID_FIELD'] else self.all_items_set
        pos_ids, neg_ids = [], []
        for i in ids:
            pos_ids_my = a2a_dict[i]
            if len(pos_ids_my) <= 0 or len(pos_ids_my)/len(all_set) > 0.8:
                pos_ids.append(0)
                neg_ids.append(0)
                continue
            pos_id = random.sample(pos_ids_my, 1)[0]
            pos_ids.append(pos_id)
            neg_id = random.sample(all_set, 1)[0]
            while neg_id in pos_ids_my:
                neg_id = random.sample(all_set, 1)[0]
            neg_ids.append(neg_id)
        return torch.tensor(pos_ids).type(torch.LongTensor), torch.tensor(neg_ids).type(torch.LongTensor)

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u

    def _get_history_users_i(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        iid_freq = self.dataset.df.groupby(iid_field)[uid_field]
        for i, u_ls in iid_freq:
            self.history_users_per_i[i] = set(u_ls.values)
        return self.history_users_per_i

class EvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr+self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr+inter_cnt].clone()
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step

        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_items_per_u(self, eval_users):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()