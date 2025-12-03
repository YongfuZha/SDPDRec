# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from diffusier import Diffuser
from Flow_Marching import FlowMatcher
from scipy.sparse import csr_matrix
class AbstractRecommender(nn.Module):
    def pre_epoch_processing(self):
        pass
     
    def post_epoch_processing(self):
        pass
     
    def calculate_loss(self, interaction):
        raise NotImplementedError
     
    def predict(self, interaction):
        raise NotImplementedError
     
    def full_sort_predict(self, interaction):
        raise NotImplementedError
    
    def __str__(self):
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()
        
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()
        
        self.batch_size = config['train_batch_size']
        self.device = config['device']
        
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
             
            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding) or isinstance(module, nn.Parameter):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
     
    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm
     
    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
     
    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    try:
        from torch_scatter import scatter_add
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    except ImportError:
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, row, edge_weight)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

class FREEDOM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config.get('lambda_coeff', 0.9)
        self.cf_model = config.get('cf_model', 'lightgcn')
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config.get('degree_ratio', 1.0)

        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix_csr = dataset.inter_matrix(form='csr').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)

        # if os.path.exists(mm_adj_file):
        #     self.mm_adj = torch.load(mm_adj_file, weights_only=True)
        # else:
        if self.v_feat is not None:
            indices, v_adj,image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.mm_adj = image_adj
        if self.t_feat is not None:
            indices, t_adj,text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            # self.ori_mm_adj = self.mm_image_weight * v_adj + (1.0 - self.mm_image_weight) * t_adj
            self.ori_mm_adj = t_adj
            # del text_adj
            # del image_adj
        self.mm_image_adj=image_adj
        self.mm_text_adj=text_adj
        torch.save(self.mm_adj, mm_adj_file)
        self.users_v_diffuser=Diffuser(self.embedding_dim,self.device)
        self.users_t_diffuser=Diffuser(self.embedding_dim,self.device)
        # self.cf_diffuser=Diffuser(self.embedding_dim,self.device)
        self.temperature=0.6
        self.re_mm_graph=None
        self.users_items_index=dataset.history_items_per_u
        self.max_len=self.get_mean_length()
        self.interaction_matrix_dense=torch.tensor(self.interaction_matrix_csr.todense()).to(self.device)
        #
        # self.ui_mm_graph=self.arg_get_norm_adj_mat().to(self.device)
        self.joint_mm_mlp=torch.nn.Linear(2*self.embedding_dim,self.embedding_dim)

        self.gate_net = torch.nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 3 * self.embedding_dim),
        )
        self.scale = torch.nn.Parameter(torch.ones(3))

        self.final_mlp=torch.nn.Sequential(
            nn.Linear(2*self.embedding_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.embedding_dim, self.embedding_dim),
        )
        self.flag=False
        self.diff_users_feats=None

        self.v_feats = self.image_trs(self.image_embedding.weight)
        self.t_feats = self.text_trs(self.text_embedding.weight)
        self.history_items_v_mean, self.history_items_v_var = self.get_full_users_interactions_feats(self.v_feats)
        self.history_items_t_mean, self.history_items_t_var = self.get_full_users_interactions_feats(self.t_feats)

        # t_feats=self.get_item_mm_emb(self.mm_text_adj, 1, self.item_id_embedding.weight)
        # v_feats=self.get_item_mm_emb(self.mm_image_adj, 1, self.item_id_embedding.weight)
        history_items = self.get_user_history_items()
        self.history_items_id_emb = self.item_id_embedding.weight[history_items].to(self.device)
        self.history_items_v_emb = self.v_feats[history_items].to(self.device)
        self.history_items_t_emb = self.t_feats[history_items].to(self.device)

        #门控融合
        self.user_gate = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.item_gate = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.begin_save=1

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj,tensor_adj=self.compute_normalized_laplacian(indices, adj_size)
        return indices, adj,tensor_adj

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return adj,torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        row_indices = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col_indices = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(row_indices), dtype=np.float32)
        
        A = sp.coo_matrix((data, (row_indices, col_indices)), 
                          shape=(self.n_users + self.n_items, self.n_users + self.n_items), 
                          dtype=np.float32).tocsr()
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def arg_get_norm_adj_mat(self):
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        row_indices = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col_indices = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(row_indices), dtype=np.float32)

        A = sp.coo_matrix((data, (row_indices, col_indices)),
                          shape=(self.n_users + self.n_items, self.n_users + self.n_items),
                          dtype=np.float32).tocsr()
        A=A.tolil()
        A[self.n_users:, self.n_users:] = self.ori_mm_adj.to_dense().cpu()
        A=A.tocsr()

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def get_item_mm_emb(self,adj,layers,emb):
        h = emb
        for i in range(layers):
            h = torch.sparse.mm(adj, h)
        return h

    def get_ui_emb(self,users_embedding,item_embedding,adj,layers):
        ego_embeddings = torch.cat((users_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def get_diff_emb(self,users_emb,history_item_feats):

        diff_mm_feats = self.multimodeal_diffuser.forward(users_emb, history_item_feats)

        # fusion
        # concat = torch.cat([diff_mm_feats, diff_cf_items], dim=-1)  #
        # gate_weight = torch.sigmoid(self.gate(concat))  #
        # fused = gate_weight * diff_cf_items + (1 - gate_weight) * diff_mm_feats

        return diff_mm_feats

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        #
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        #diff
        x_users_emb = self.user_embedding.weight.to(self.device)


        history_items_v_mean=self.history_items_v_mean
        history_items_v_var=self.history_items_v_var
        history_items_t_mean=self.history_items_t_mean
        history_items_t_var=self.history_items_t_var
        history_items_id_emb=self.history_items_id_emb
        history_items_v_emb=self.history_items_v_emb
        history_items_t_emb=self.history_items_t_emb
        users_feats = self.users_t_diffuser.forward(history_items_v_mean.detach(),history_items_v_var.detach(),history_items_t_mean.detach(),history_items_t_var.detach(),
            x_users_emb, history_items_id_emb.detach(),history_items_v_emb.detach(),history_items_t_emb.detach())
        #


        # u_g_embeddings = ua_embeddings[users]
        # u_g_embeddings = ua_embeddings[users]
        # pos_i_g_embeddings = ia_embeddings[pos_items]
        # neg_i_g_embeddings = ia_embeddings[neg_items]

        #neg
        # diff_users_feats=torch.cat([users_v_feats,users_t_feats],dim=1)
        # diff_users_f_emb=self.final_mlp(diff_users_feats)
        # diff_users_f_emb=0.1*users_v_feats+0.9*users_t_feats

        # items_emb=self.item_id_embedding.weight
        items_emb=self.item_id_embedding.weight*(1*self.v_feats.detach())*(1*self.t_feats.detach())
        diff_users_emb,diff_items_emb=self.get_ui_emb(users_feats,items_emb.to(self.device),self.norm_adj,self.n_ui_layers)




        u_g_embeddings=ua_embeddings[users]

        # u_g_embeddings = diff_users_emb[users] + ua_embeddings[users]
        # pos_i_g_embeddings = ia_embeddings[pos_items]
        # neg_i_g_embeddings = ia_embeddings[neg_items]

        # user_neg_diff_items=self.sample_neg_items(users,pos_items,ia_embeddings,ua_embeddings[users])
        user_neg_diff_items=self.sample_neg_items(users,pos_items,diff_items_emb,diff_users_emb[users])

        # fusion
        # user_concat = torch.cat([ua_embeddings, diff_users_emb], dim=-1)  # (batch_size, 2*dim)
        # gate_weight = torch.sigmoid(self.user_gate(user_concat))  # (batch_size, dim)
        # user_fused = gate_weight * ua_embeddings + (1 - gate_weight) * diff_users_emb
        #
        # item_concat = torch.cat([ia_embeddings, diff_items_emb], dim=-1)  # (batch_size, 2*dim)
        # gate_weight = torch.sigmoid(self.item_gate(item_concat))  # (batch_size, dim)
        # item_fused = gate_weight * ia_embeddings + (1 - gate_weight) * diff_items_emb


        # user_neg_diff_items_embedding=ia_embeddings[user_neg_diff_items]
        user_neg_bpr_loss=self.bpr_loss(u_g_embeddings, ia_embeddings[pos_items], ia_embeddings[user_neg_diff_items])

        # item_neg_diff_items = self.sample_neg_items(users,pos_items,ia_embeddings,ua_embeddings[users])
        # item_neg_diff_items_embedding = ia_embeddings[item_neg_diff_items]
        # item_neg_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, item_neg_diff_items_embedding)

        diff_bpr_loss=user_neg_bpr_loss


        # fused=fused+u_g_embeddings
        #loss
        # u_g_embeddings+=0.1*(diff_users_v_feats+diff_users_t_feats)

        users_emb,items_emb=self.get_ui_emb(self.user_embedding.weight,self.item_id_embedding.weight,self.norm_adj,self.n_ui_layers)
        cl_loss=0
        users_cl_loss=self.diffusion_duibi(diff_users_emb[users],users_emb[users])
        items_cl_loss=self.diffusion_duibi(diff_items_emb[pos_items],items_emb[pos_items])
        cl_loss=users_cl_loss+items_cl_loss

        # batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)+diff_loss+0.01*users_cl_loss+diff_bpr_loss
        diff_loss = F.mse_loss(x_users_emb[users], users_feats[users])
        batch_mf_loss = self.bpr_loss(u_g_embeddings, ia_embeddings[pos_items], ia_embeddings[neg_items])+0.5*diff_bpr_loss+0.1*diff_loss+0.001*cl_loss
        # batch_mf_loss = self.bpr_loss(u_g_embeddings, ia_embeddings[pos_items], ia_embeddings[neg_items])

        mf_v_loss, mf_t_loss = 0.0, 0.0
        # if self.t_feat is not None:
        #     text_feats = self.text_trs(self.text_embedding.weight)
        #     mf_t_loss = self.bpr_loss(diff_users_emb[users], text_feats[pos_items], text_feats[neg_items])
        # if self.v_feat is not None:
        #     image_feats = self.image_trs(self.image_embedding.weight)
        #     mf_v_loss = self.bpr_loss(diff_users_emb[users], image_feats[pos_items], image_feats[neg_items])
        return batch_mf_loss + 0.001 * (mf_t_loss + mf_v_loss)

    #
    # def get_user_history_items_emb(self,users,items_v_features,items_t_features):
    #     history_items_v_emb=[]
    #     history_items_t_emb=[]
    #     for user in users:
    #         history_items=self.get_user_padding_items(user)
    #         history_items_v_emb.append(items_v_features[history_items])
    #         history_items_t_emb.append(items_t_features[history_items])
    #     history_items_v_emb=torch.stack(history_items_v_emb,dim=0)
    #     history_items_t_emb=torch.stack(history_items_t_emb,dim=0)
    #     return history_items_v_emb,history_items_t_emb

    def get_user_history_items(self):
        history_items_all=[]
        for user,items in self.users_items_index.items():
            history_items=self.get_user_padding_items(user)
            history_items_all.append(history_items)
        history_items=torch.LongTensor(history_items_all)
        return history_items

    def get_full_users_interactions_feats(self,items_features):
        history_items_emb_mean = []
        history_items_emb_var=[]
        for user,items in self.users_items_index.items():
            items=list(items)
            emb_mean=torch.mean(items_features[items],dim=0)
            emb_var=torch.var(items_features[items],dim=0)
            emb_var=torch.clamp(emb_var, min=1e-6)
            history_items_emb_mean.append(emb_mean)
            history_items_emb_var.append(emb_var)

        history_items_emb_mean=torch.stack(history_items_emb_mean,dim=0)
        history_items_emb_var=torch.stack(history_items_emb_var,dim=0)
        return history_items_emb_mean,history_items_emb_var

    def sample_neg_items_items(self,users,pos_items,diff_item_embeddings):
        sample_k=0.1
        pos_diff_item_embeddings=diff_item_embeddings[pos_items]


        num_sample=int(0.1*diff_item_embeddings.shape[0])
        random_indices=torch.randint(0,diff_item_embeddings.shape[0],(num_sample,)).to(self.device)
        dot_products = torch.matmul(pos_diff_item_embeddings, diff_item_embeddings[random_indices].t())
        interaction_matrix=self.interaction_matrix_dense[users][:,random_indices]
        dot_products[interaction_matrix==1]=float("-inf")

        _,top_indices=torch.topk(dot_products,k=int(sample_k*dot_products.shape[1]),dim=1)
        random_ids=torch.randint(0,top_indices.shape[1],(len(pos_items),))
        most_similar_ids=top_indices[torch.arange(len(pos_items)),random_ids]

        return random_indices[most_similar_ids]

    def sample_neg_items(self, users, pos_items, diff_item_embeddings, user_embeddings=None):
        """
        users: [batch_size]
        pos_items: [batch_size]
        diff_item_embeddings: [num_items, dim]
        user_embeddings: [batch_size, dim]  ← 新增参数
        """
        sample_k = 0.1
        num_sample = int(0.1 * diff_item_embeddings.shape[0])

        # 随机采样候选物品池
        random_indices = torch.randint(0, diff_item_embeddings.shape[0], (num_sample,)).to(self.device)
        candidate_embs = diff_item_embeddings[random_indices]  # [num_sample, dim]

        if user_embeddings is None:
            # 如果没传 user_embeddings，退化为原方法：用正样本物品的 embedding
            pos_diff_item_embeddings = diff_item_embeddings[pos_items]
            dot_products = torch.matmul(pos_diff_item_embeddings, candidate_embs.t())  # [batch, num_sample]
        else:
            # 使用用户 embedding 计算相似度 → 更个性化
            dot_products = torch.matmul(user_embeddings, candidate_embs.t())  # [batch, num_sample]

        # 屏蔽用户已经交互过的物品
        interaction_matrix = self.interaction_matrix_dense[users][:, random_indices]  # [batch, num_sample]
        dot_products[interaction_matrix == 1] = float("-inf")

        # 选出 top-k 最相似的难负样本
        _, top_indices = torch.topk(dot_products, k=int(sample_k * num_sample), dim=1)  # [batch, k]

        # 每个用户从 top-k 中随机选一个（增加多样性）
        random_ids = torch.randint(0, top_indices.shape[1], (len(pos_items),)).to(self.device)
        most_similar_ids = top_indices[torch.arange(len(pos_items)), random_ids]  # [batch]

        # 返回全局物品 ID
        return random_indices[most_similar_ids]  # [batch]

    def full_sort_predict(self, interaction):
        user = interaction[0]
        self.flag = False



        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        #diff
        # history_items_v_mean = self.history_items_v_mean
        # history_items_v_var = self.history_items_v_var
        # history_items_t_mean = self.history_items_t_mean
        # history_items_t_var = self.history_items_t_var
        # history_items_id_emb = self.history_items_id_emb
        # history_items_v_emb = self.history_items_v_emb
        # history_items_t_emb = self.history_items_t_emb
        # users_feats = self.users_t_diffuser.reverse(history_items_v_mean, history_items_v_var, history_items_t_mean,
        #                                             history_items_t_var,
        #                                             restore_user_e, history_items_id_emb.detach(),
        #                                             history_items_v_emb.detach(), history_items_t_emb.detach())
        #
        # diff_users_emb, diff_items_emb = self.get_ui_emb(users_feats, self.item_id_embedding.weight.to(self.device),
        #                                                  self.norm_adj, self.n_ui_layers)

        # fusion
        # user_concat = torch.cat([u_embeddings, diff_users_emb[user]], dim=-1)  # (batch_size, 2*dim)
        # gate_weight = torch.sigmoid(self.user_gate(user_concat))  # (batch_size, dim)
        # user_fused = gate_weight * u_embeddings + (1 - gate_weight) * diff_users_emb[user]
        #
        # item_concat = torch.cat([restore_item_e, diff_items_emb], dim=-1)  # (batch_size, 2*dim)
        # gate_weight = torch.sigmoid(self.item_gate(item_concat))  # (batch_size, dim)
        # item_fused = gate_weight * restore_item_e + (1 - gate_weight) * diff_items_emb
        # u_embeddings = user_fused

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def diffusion_duibi(self,view1,view2):
        user_emb1 = view1
        user_emb0 = view2
        normalize_user_emb1 = torch.norm(user_emb1, p=2, dim=1)
        normalize_user_emb0 = torch.norm(user_emb0, p=2, dim=1)

        normalize_user_emb1 = normalize_user_emb1.view(len(normalize_user_emb1), -1)
        normalize_user_emb0 = normalize_user_emb0.view(len(normalize_user_emb0), -1)
        final_user_emb1 = user_emb1 / normalize_user_emb1
        final_user_emb0 = user_emb0 / normalize_user_emb0

        pos_score_user = torch.sum(torch.multiply(final_user_emb1, final_user_emb0), dim=1)
        # ttl_score_user = torch.matmul(final_user_emb1, final_user_emb0.T) - torch.diag(pos_score_user)
        ttl_score_user = torch.matmul(final_user_emb1, final_user_emb0.T)

        pos_score_user = torch.exp(pos_score_user / self.temperature)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.temperature), dim=1)

        ssl_loss_user1 = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        ssl_loss = ssl_loss_user1 / (len(view1))

        return ssl_loss

    # def diffusion_duibi(self, view1, view2):
    #     z1 = F.normalize(view1, p=2, dim=1)
    #     z2 = F.normalize(view2, p=2, dim=1)
    #
    #     logits = torch.matmul(z1, z2.T) / self.temperature
    #     labels = torch.arange(len(logits), device=logits.device)
    #
    #     loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    #     return loss / 2.0

    def my_duibi(self, users_data, users_emb,id_mm_s_embedding):

        neighor_seqs = []
        for user in users_data:
            neighor_seqs.append(self.get_user_padding_items(user))

        neighor_emb=id_mm_s_embedding[neighor_seqs]
        neighor_emb = torch.mean(neighor_emb, dim=1)

        user_emb1 = users_emb
        user_emb0 = neighor_emb
        normalize_user_emb1 = torch.norm(user_emb1, p=2, dim=1)
        normalize_user_emb0 = torch.norm(user_emb0, p=2, dim=1)
        #
        normalize_user_emb1 = normalize_user_emb1.view(len(normalize_user_emb1), -1)
        normalize_user_emb0 = normalize_user_emb0.view(len(normalize_user_emb0), -1)
        final_user_emb1 = user_emb1 / normalize_user_emb1
        final_user_emb0 = user_emb0 / normalize_user_emb0

        pos_score_user = torch.sum(torch.multiply(final_user_emb1, final_user_emb0), dim=1)
        # ttl_score_user = torch.matmul(final_user_emb1, final_user_emb0.T) - torch.diag(pos_score_user)
        ttl_score_user = torch.matmul(final_user_emb1, final_user_emb0.T)

        pos_score_user = torch.exp(pos_score_user / self.temperature)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.temperature), dim=1)

        ssl_loss_user1 = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        ssl_loss = ssl_loss_user1 / (len(user_emb1))

        return ssl_loss

    def get_user_padding_items(self, user):
        if user == 0:
            return [0] * self.max_len
        neighor = self.users_items_index[user]
        neighor_length = len(neighor)
        neighor=list(neighor)

        if neighor_length < self.max_len:
            neighor = np.array(neighor)
            neighor = np.random.choice(neighor, self.max_len).tolist()
        else:
            neighor = np.array(neighor)
            neighor = np.random.choice(neighor, self.max_len, replace=False).tolist()
        return neighor

    def get_mean_length(self):
        mean_length=0
        total_users=0
        for user,items in self.users_items_index.items():
            mean_length+=len(items)
            if len(items)>0:
                total_users+=1
        return int(mean_length/total_users)

    def get_huihe_mm_graph_v2(self,mm_feats):
        k=20
        mm_adj = self.get_zuhe_image_text_graph(mm_feats, k)

        graph = self._convert_sp_mat_to_sp_tensor(mm_adj).to(self.device)
        # graph=0.3*graph+0.7*self.mm_adj
        return graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_zuhe_image_text_graph(self, mm_info, k):
        mm_info = mm_info.div(torch.norm(mm_info, p=2, dim=-1, keepdim=True))
        image_sim_items_cs = torch.mm(mm_info, mm_info.T)
        image_sim_items_sort = torch.argsort(image_sim_items_cs)
        image_sim_items_sort = image_sim_items_sort[:, -k - 2:-2]
        # image_sim_items_sort = image_sim_items_sort[:, :k]

        train_b_items = []
        train_e_items = []
        # xishu=[]
        for i in range(1, len(image_sim_items_sort)):
            uid = i
            items = image_sim_items_sort[i].tolist()
            train_b_items.extend([uid] * len(items))
            train_e_items.extend(items)
            # xishu.extend(image_sim_items_cs[i][image_sim_items_sort[i]])

        item_item_net = csr_matrix((np.ones(len(train_b_items)), (train_b_items, train_e_items)),
                                   shape=(self.n_items, self.n_items))

        adj_mat = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        I_I = item_item_net.tolil()
        adj_mat[:, :] = I_I
        adj_mat = adj_mat.todok()

        rowsum = 1e-7 + np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        return norm_adj

    def get_user_items(self):
        train_file = "./data/baby" + '/train.txt'
        users_items_index = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    #
                    uid = int(l[0])
                    if uid not in users_items_index:
                        users_items_index[uid] = []
                    users_items_index[uid].extend(items)
        return users_items_index

class LightGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN_Encoder, self).__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.n_layers = 3 if config.get('n_layers') is None else config['n_layers']
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix.tolil()
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users), [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col), [1]*inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        users, items = inputs[0], inputs[1]
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings