import gol
import torch
import torch.nn as nn
from torchsde import sdeint
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class BiGraphEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(BiGraphEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BiGAT(hid_dim)

    def encode(self, embs, G_u):
        return self.encoder(embs, G_u)

'''
BiGAT: Bi-directional Graph Convolution, is as a part of Direction-aware POI Transition Graph Learning
    Input: User-specific POI transition graph $ \mathcal{G}_u $
    Return: Node Feature of the POI transition graph $ \tilde{H}_u $ 
'''
class BiGAT(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(BiGAT, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)  # alpha_src, alpha_dst
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        
        # attention_weight
        self.attention_weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.attention_weight.data)
        
        self.time_scaling = nn.Parameter(torch.Tensor([0.5])) 
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, G_u):
        POI_embs, delta_dis_embs, delta_time_embs = embs
        time = G_u.x[:, 1].unsqueeze(1).float()
        mean = time.mean()
        std = time.std()
        time = (time - mean) / std

        sess_idx = G_u.x[:, 0].squeeze()
        in_degree = G_u.edge_index[0].bincount(minlength=G_u.num_nodes).float()
        out_degree = G_u.edge_index[1].bincount(minlength=G_u.num_nodes).float()
        edge_index = G_u.edge_index
        edge_time  = G_u.edge_time
        edge_dist  = G_u.edge_dist
        edge_in_degree = in_degree[edge_index[1]]
        edge_out_degree = out_degree[edge_index[0]]
        x = POI_embs[sess_idx] * (1 + self.time_scaling * time)
        
        edge_l = delta_dis_embs[edge_dist]
        edge_t = delta_time_embs[edge_time]
        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        
        H_u = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_in_degree=edge_in_degree,
                             edge_out_degree=edge_out_degree, edge_size=edge_index.size(1))

        return H_u

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_in_degree, edge_out_degree, edge_size):

        attention_coefficients = torch.matmul(x_i[edge_size:] + edge_l + edge_t, self.attention_weight.t())

        src_attention = self.alpha_src(attention_coefficients[:edge_size]).squeeze(-1) * edge_out_degree[:edge_size]
        dst_attention = self.alpha_dst(attention_coefficients[:edge_size]).squeeze(-1) * edge_in_degree[:edge_size]
        
        # softmax on tot_attention
        tot_attention = torch.cat((src_attention, dst_attention), dim=0)
        attn_weight = softmax(tot_attention, edge_index_i)

        # attn_weight on neighbor node features
        updated_rep = x_j * attn_weight.unsqueeze(-1)
        return updated_rep
    
    def update(self, aggr_out, x):
        return aggr_out + x 
    

'''
SDEsolver: Stochastic Differential Equation Solver 
'''
class SDEsolver(nn.Module):
    sde_type = 'stratonovich'   # available: {'ito', 'stratonovich'}
    noise_type = 'scalar'       # available: {'general', 'additive', 'diagonal', 'scalar'}

    def __init__(self, f, g):
        super(SDEsolver).__init__()
        self.f, self.g = f, g

    def f(self, t, y): 
        return self.f(t, y)
    
    def g(self, t, y): 
        return self.g(t, y)

'''
Bridge_Diffusion: Bridge-based Diffusion POI Generation model
'''
class Bridge_Diffusion(nn.Module):
    def __init__(self, hid_dim, beta_min, beta_max, dt):
        super(Bridge_Diffusion, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt
        self.sigma_data_end = 0.5
        self.sigma_data = 0.5
        self.cov_xy = 0.
        self.c = 1

        self.time_embed = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # score-based neural network stacked multiple fully connected layers
        self.score_FC = nn.Sequential(
                    nn.Linear(3 * hid_dim, 3 * hid_dim),
                    nn.BatchNorm1d(3 * hid_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.2), 
                    nn.Linear(3 * hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.2), 
                    nn.Linear(hid_dim, hid_dim)
                    )
        
        for w in self.score_FC:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    # a time-dependent score-based neural network to estimate marginal probability
    def Est_score(self, c_in, x, x_T, condition):
        return self.score_FC(torch.cat((x * c_in, x_T * c_in, condition * c_in), dim=-1))

    def vp_mean_std_snr(self, t):
        vp_logs = -0.25 * (t+0.0001) ** 2 * (self.beta_max - self.beta_min) - 0.5 * (t+0.0001) * self.beta_min  # ln(α_t) = -0.5*B, B=0.5*(beta_max-beta_min)*t^2 + beta_min*t
        vp_logsnr = - torch.log((0.5 * (self.beta_max - self.beta_min) * ((t+0.0001) ** 2) + self.beta_min * (t+0.0001)).exp() - 1) # ln(SNR)= -ln(e^B-1)
        return vp_logs, vp_logsnr
    
    # unified modeling with  four scaling functions
    def get_bridge_scalings(self, t, T):
        logs_t, logsnr_t = self.vp_mean_std_snr(t)
        logs_T, logsnr_T = self.vp_mean_std_snr(T)
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()
        A = a_t ** 2 * self.sigma_data_end ** 2 + b_t ** 2 * self.sigma_data ** 2 + 2 * a_t * b_t * self.cov_xy + self.c ** 2 * c_t
        weightings = A / (a_t ** 2 * (self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * c_t)
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data ** 2 + a_t * self.cov_xy) / A
        c_out = (a_t ** 2 * (self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * c_t) ** 0.5 * c_in
        return c_skip, c_out, c_in, weightings
    
    # score-based Denoiser 
    def D_doise(self, x, x_T, condition, t, T):
        c_skip, c_out, c_in, weightings = [append_dims(x, x_T.ndim) for x in self.get_bridge_scalings(t, T)]

        model_output = self.Est_score(c_in, x, x_T, condition)
        denoised = c_out * model_output + c_skip * x
        return model_output, denoised, weightings
    
    #  Define the drift term f and diffusion term g of Forward SDE
    def ForwardSDE_diff(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(gol.device)
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def ReverseSDE_gener(self, x, condition, T):
        x_T = x.clone()
        def get_beta_t(_t):
            beta_t_1 = self.beta_min + _t * (self.beta_max - self.beta_min)
            beta_t_2 = self.beta_min + (self.beta_max - self.beta_min) * torch.sin(torch.pi / 2 * _t) ** 2
            beta_t_3 = self.beta_min * torch.exp(torch.log(torch.tensor(self.beta_max / self.beta_min)) * _t)
            beta_t_4 = self.beta_min + _t * (self.beta_max - self.beta_min) ** 2
            beta_t_5 = 0.1 * torch.exp(6 * _t)
            return beta_t_1

        # drift term f(): {_t: current time point, y: current state, returns the value of the drift term}
        def f(_t, y):
            beta_t = get_beta_t(T-_t)
            logs_t, logsnr_t = self.vp_mean_std_snr(T-_t)
            logs_T, logsnr_T = self.vp_mean_std_snr(T)
            # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2

            h = -(y - torch.exp(logs_t - logs_T) * x_T) / ((logs_t - logsnr_t / 2).exp()) ** 2 / torch.expm1(logsnr_t - logsnr_T)

            _, score, _ = self.D_doise(y, x_T, condition, T-_t, T)

            s = (y - (logsnr_T - logsnr_t + logs_t - logs_T).exp() * x_T + -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp() * score ) \
                / ((logs_t - logsnr_t / 2).exp()) ** 2 / (-torch.expm1(logsnr_T - logsnr_t))
            
            ## score = self.score_FC(y)
            drift = -0.5 * beta_t * y - beta_t * (s - h)
            return drift

        # diffusion term g(): {_t: current time point, y: current state, returns the value of the diffusion term}
        def g(_t, y):
            beta_t = get_beta_t(T-_t).unsqueeze(-1)
            bs = y.size(0)
            
            # noise Tensors [bs, self.hid_dim, 1] = [1024, 64, 1]  with all elements of 1
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion

        def g_diagonal_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim), device=y.device)
            diagonal_beta = torch.diag(beta_t * torch.ones(dim, device=y.device))
            diffusion = (diagonal_beta ** 0.5).mm(noise.t()).t()
            return diffusion + y

        def g_vector_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim, brownian_size = y.size(0), y.size(1), y.size(1)
            noise = torch.randn((bs, dim, brownian_size), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion

        def g_full_cov_noise_3d(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim, dim), device=y.device)
            covariance_matrix = torch.eye(dim, device=y.device)
            covariance_matrix = covariance_matrix * beta_t
            cholesky_matrix = torch.linalg.cholesky(covariance_matrix)
            diffusion = torch.einsum('bij,jk->bik', noise, cholesky_matrix)
            return diffusion

        ts = torch.Tensor([0.001,1]).to(gol.device)

        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def marginal_prob(self, x, x_T, t, T):

        # print(t)
        logs_t, logsnr_t = self.vp_mean_std_snr(t)
        logs_T, logsnr_T = self.vp_mean_std_snr(T)
        # print(T)
        # print(logs_T)
        # print(logsnr_T)
        # print('fvfvf',logsnr_T - logsnr_t + logs_t - logs_T)
        # print('logs_t:',logs_t)
        # print('logsnr_t:',logsnr_t)
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = (-torch.expm1(logsnr_T - logsnr_t) * logs_t.exp())
        # print('a_t:',a_t)
        # print('b_t',b_t)
        std = ((-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp())
        mean = a_t * x_T + b_t * x
        return mean, std

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def timestep_embedding(timesteps, dim, max_period=10000):
    if isinstance(timesteps, float):
        timesteps = torch.tensor([timesteps], dtype=torch.float32)
    half_dim = dim // 2
    freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * math.log(max_period) / half_dim)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding