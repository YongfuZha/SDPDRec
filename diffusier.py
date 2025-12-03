
import torch

from torch import nn
import numpy as np
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import math
import utils
import scipy.sparse as sp


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Diffuser(nn.Module):
    def __init__(self, hidden_size,device):
        super(Diffuser, self).__init__()
        self.timesteps = 32
        self.beta_start = 0.001
        self.beta_end = 0.02
        self.hidden_size=hidden_size
        self.dev=device
        self.beta_sche="cosine"

        # self.w = w

        if self.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.w_q = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.init(self.w_q)
        self.w_k = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.init(self.w_k)
        self.w_v = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.init(self.w_v)
        self.ln = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.ffn=PointWiseFeedForward(self.hidden_size, dropout_rate=0.2)


    def init(self,m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,torch.nn.Parameter):
            torch.nn.init.xavier_normal_(m)

    def forward(self, history_items_v_mean,history_items_v_var,history_items_t_mean,history_items_t_var,
                users_emb,history_items_id_emb,history_items_v_emb,history_items_t_emb):

        users_visual_noise = torch.randn_like(users_emb).to(self.dev)
        users_text_noise = torch.randn_like(users_emb).to(self.dev)
        users_visual_noise=history_items_v_mean+torch.sqrt(torch.clamp(history_items_v_var,min=1e-6))*users_visual_noise
        users_text_noise=history_items_t_mean+torch.sqrt(torch.clamp(history_items_t_var,min=1e-6))*users_text_noise


        times_info = torch.randint(0, self.timesteps, (len(users_emb),), device=self.dev).long()
        users_noise=0.1*users_visual_noise+0.9*users_text_noise
        # users_noise = torch.randn_like(users_emb).to(self.dev)
        users_f = self.q_sample(users_emb, times_info, users_noise)
        # users_t_f = self.q_sample(users_emb, times_info, users_text_noise)

        times_info_embedding = self.get_time_s(times_info)

        diffu_log_feats=torch.cat([times_info_embedding.unsqueeze(1),history_items_id_emb,history_items_v_emb,history_items_t_emb
                                      ,users_f.unsqueeze(1)],dim=1)

        users_final_emb=self.selfAttention(diffu_log_feats)

        return users_final_emb

    # def predict(self, user_ids, log_seqs, item_indices): # for inference
    # def reverse(self, users_emb,condition_emb):  # for inference
    #
    #     users_visual_noise = torch.randn_like(users_emb).to(self.dev)
    #     users_text_noise = torch.randn_like(users_emb).to(self.dev)
    #     #x_traget->x_i_v and x_source->x_i_t
    #     x_i_v = users_visual_noise
    #     x_i_t = users_text_noise
    #     i_inter_final_feats=condition_emb
    #
    #     # source_info=torch.nn.functional.normalize(source_info,p=2,dim=-1)
    #     # mix_info=torch.nn.functional.normalize(mix_info,p=2,dim=-1)
    #     #
    #     for i in reversed(range(0, self.timesteps)):
    #         t = torch.tensor([i] * x_i_v.shape[0], dtype=torch.long).to(self.dev)
    #         # -----------------------
    #         times_info_embeddings = self.get_time_s(t)
    #
    #         # --
    #         x_t_i_v = x_i_v
    #         x_t_i_t = x_i_t
    #         # x+=+times_info_embeddings
    #         diffu_log_feats = torch.cat([times_info_embeddings.unsqueeze(1),i_inter_final_feats,x_i_v.unsqueeze(1),x_i_t.unsqueeze(1)], dim=1)
    #
    #         diffu_r_x_v,diffu_r_x_t=self.selfAttention(diffu_log_feats)
    #         # --
    #
    #         model_mean_i_v = (
    #                 self.extract(self.posterior_mean_coef1, t, x_t_i_v.shape) * diffu_r_x_v +
    #                 self.extract(self.posterior_mean_coef2, t, x_t_i_v.shape) * x_i_v
    #         )
    #         model_mean_i_t = (
    #                 self.extract(self.posterior_mean_coef1, t, x_t_i_t.shape) * diffu_r_x_t +
    #                 self.extract(self.posterior_mean_coef2, t, x_t_i_t.shape) * x_i_t
    #         )
    #
    #         if i == 0:
    #             x_i_v = model_mean_i_v
    #             x_i_t = model_mean_i_t
    #         else:
    #             # ---
    #             posterior_variance_v = self.extract(self.posterior_variance, t, x_i_v.shape)
    #             posterior_variance_t = self.extract(self.posterior_variance, t, x_i_t.shape)
    #             noise_i_v = torch.randn_like(x_i_v)
    #             noise_i_t = torch.randn_like(x_i_t)
    #
    #             x_i_v = model_mean_i_v + torch.sqrt(posterior_variance_v) * noise_i_v
    #             x_i_t = model_mean_i_t + torch.sqrt(posterior_variance_t) * noise_i_t
    #             # x = model_mean
    #     #---
    #
    #     return x_i_v,x_i_t

    def reverse(self, history_items_v_mean,history_items_v_var,history_items_t_mean,history_items_t_var,
                users_emb,history_items_id_emb,history_items_v_emb,history_items_t_emb):  # for inference

        users_visual_noise = torch.randn_like(users_emb).to(self.dev)
        users_text_noise = torch.randn_like(users_emb).to(self.dev)
        users_visual_noise=history_items_v_mean+torch.sqrt(torch.clamp(history_items_v_var,min=1e-6))*users_visual_noise
        users_text_noise=history_items_t_mean+torch.sqrt(torch.clamp(history_items_t_var,min=1e-6))*users_text_noise
        #
        users_noise=0.1*users_visual_noise+0.9*users_text_noise
        # users_noise=torch.randn_like(users_emb).to(self.dev)
        #x_traget->x_i_v and x_source->x_i_t
        x_i = users_noise

        # source_info=torch.nn.functional.normalize(source_info,p=2,dim=-1)
        # mix_info=torch.nn.functional.normalize(mix_info,p=2,dim=-1)
        #
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * x_i.shape[0], dtype=torch.long).to(self.dev)
            # -----------------------
            times_info_embeddings = self.get_time_s(t)

            # --
            x_t_i = x_i
            # x+=+times_info_embeddings
            diffu_log_feats = torch.cat([times_info_embeddings.unsqueeze(1),history_items_id_emb,history_items_v_emb,history_items_t_emb
                                            ,x_t_i.unsqueeze(1)], dim=1)

            diffu_r_x=self.selfAttention(diffu_log_feats)
            # --

            model_mean_i = (
                    self.extract(self.posterior_mean_coef1, t, x_t_i.shape) * diffu_r_x +
                    self.extract(self.posterior_mean_coef2, t, x_t_i.shape) * x_i
            )

            if i == 0:
                x_i = model_mean_i
            else:
                # ---
                posterior_variance_v = self.extract(self.posterior_variance, t, x_i.shape)
                noise_i = torch.randn_like(x_i)

                x_i = model_mean_i + torch.sqrt(posterior_variance_v) * noise_i
                # x = model_mean
        #---

        return x_i

    def selfAttention(self,features):
        features = self.ln(features)  # [B, L, D]

        # Get Q, K, V
        q = self.w_q(features)  # [B, L, D]
        k = self.w_k(features)  # [B, L, D]
        v = self.w_v(features)  # [B, L, D]

        # Scale Q
        q = q * (self.hidden_size ** -0.5)  # [B, L, D]

        # Compute attention scores: [B, L, L]
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [B, L, L]
        attn_weights = torch.softmax(attn_weights, dim=-1)  # softmax over keys (dim=-1)

        # Apply attention to V: [B, L, D]
        features = torch.matmul(attn_weights, v)  # [B, L, D]

        # features=self.ffn(features)

        # Return the last token of each sequence in the batch
        return features[:, -1, :]

    def get_time_s(self, time):
        half_dim = self.hidden_size // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.dev) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def extract(self, a, t, x_shape):
        # res = a.to(device=t.device)[t].float()
        # while len(res.shape) < len(x_shape):
        #     res = res[..., None]
        # return res.expand(x_shape)
        # batch_size = t.shape[0]
        # out = a.gather(-1, t.cpu())
        # return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.dev)
        res = a.to(device=t.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def r_extract(self, a, t, x_shape):
        res = torch.from_numpy(a).to(device=t.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # mean=torch.mean(x_start,dim=0)
        # noise=mean+(x_start - mean).pow(2).mean().sqrt()*noise
        # return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise*torch.sign(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, x, t, t_index, diffuer_mlp):
        times_info = torch.tensor([t_index] * x.shape[0], dtype=torch.long).to(self.dev)
        times_info_embeddings = self.get_time_s(times_info)
        x_start = diffuer_mlp(torch.cat([x, times_info_embeddings], dim=-1))
        x_t = x

        model_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def denoise(self, times_info, x_T, diffuer_mlp):
        # x = self.q_sample(x, times_info)
        # x=torch.randn_like(condtion_info)
        for i in reversed(range(0, self.timesteps)):
            x_T = self.p_sample(x_T, torch.full((times_info.shape[0],), i, device=self.dev, dtype=torch.long), i,diffuer_mlp)
        return x_T

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, seqs_emb, x_t, t,seqs):
        times_info_embeddings = self.get_time_s(t)
        x_t+=times_info_embeddings
        lambda_uncertainty = torch.normal(mean=torch.full(seqs_emb.shape, self.lambda_uncertainty),
                                    std=torch.full(seqs_emb.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        seqs_emb = seqs_emb + lambda_uncertainty * x_t.unsqueeze(1)

        model_output = self.SASModel(seqs_emb, seqs)

        x_0 = model_output[:,-1,:]  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = self.r_extract(model_log_variance, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t,
                                                    t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def pp_sample(self, item_rep, noise_x_t, t,seqs):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t,seqs)
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        # sample_xt = model_mean  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick

        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t,seqs):
        device = self.dev
        indices = list(range(self.timesteps))[::-1]

        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * item_rep.shape[0], device=device)
            with torch.no_grad():
                noise_x_t = self.pp_sample(item_rep, noise_x_t, t,seqs)
        # t = th.tensor([1] * item_rep.shape[0], device=device)
        # model_output, _ = self.xstart_model(item_rep, noise_x_t, self._scale_timesteps(t), mask_seq)
        # return model_output
        return noise_x_t