import gol
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from layers import Bridge_Diffusion, BiGraphEncoder

'''PFFN: Point-wise Feed Forward Network'''
class PFFN(nn.Module):
    def __init__(self, hid_size, dropout_rate):
        super(PFFN, self).__init__()
        self.conv1 = nn.Conv1d(hid_size, hid_size, kernel_size=1) 
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hid_size, hid_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class Model(nn.Module):
    def __init__(self, n_user, n_poi):
        super(Model, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000

        # Initialize all parameters
        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.delta_dis_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.delta_time_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.delta_dis_embs)
        nn.init.xavier_normal_(self.delta_time_embs)

        self.seq_Rep = BiGraphEncoder(self.hid_dim)
        self.SDEdiff = Bridge_Diffusion(self.hid_dim, beta_min=gol.conf['beta_min'], beta_max=gol.conf['beta_max'], dt=gol.conf['dt'])

        self.CEloss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=gol.conf['dp']) 

        # Initialize Direction-aware POI Transition Graph Learning
        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.seq_PFFN = PFFN(self.hid_dim, 0.2)

        # Initialize Transformer-based User Interest Distribution Generator
        self.Trans_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.Trans_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.Trans_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.Trans_PFFN = PFFN(self.hid_dim, 0.2)


    '''
    POIGraphRep: Direction-aware POI Transition Graph Learning
        Input: User-specific POI transition graph $ \mathcal{G}_u $
        Return: User Personalized Transition Representation $ \mathcal{R}_u $
    '''
    def POIGraphRep(self, POI_embs, G_u):
        seq_embs = self.seq_Rep.encode((POI_embs, self.delta_dis_embs, self.delta_time_embs), G_u)

        if gol.conf['dropout']:
            seq_embs = self.dropout(seq_embs)

        seq_lengths = torch.bincount(G_u.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)  
        pad_mask = Seq_MASK(seq_lengths)  
        
        # Q, K, V = [seq_length, batch_size, embed_dim] 
        Q = self.seq_layernorm(seq_embs_pad)                                                                    
        K = seq_embs_pad
        V = seq_embs_pad
        output, att_weights = self.seq_attn(Q, K, V, key_padding_mask=~pad_mask)

        output = output + Q
        output = self.seq_attn_layernorm(output)

        # PFNN
        output = self.seq_PFFN(output)           
        output = [seq[:seq_len] for seq, seq_len in zip(output, seq_lengths)]

        R_u = torch.stack([seq.mean(dim=0) for seq in output], dim=0)
        return R_u


    '''
    InterestGenerator: Transformer-based User Interest Distribution Generator
        Input: User Personalized Transition Representation $ \mathcal{R}_u $, global POI embedding matrix E_P
        Return: User Personalized Interest Distribution Z_u
    '''
    def InterestGenerator(self, POI_embs, seqs, R_u):
        E_P = POI_embs

        if gol.conf['dropout']:
            E_P = self.dropout(E_P)
        
        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        E_P_seq = [E_P[seq] for seq in seqs]

        E_P_pad = pad_sequence(E_P_seq, batch_first=True, padding_value=0)
        pad_mask = Seq_MASK(seq_lengths)
        Q = self.Trans_layernorm(R_u.detach().unsqueeze(1))
        K = E_P_pad
        V = E_P_pad

        output, att_weights = self.Trans_attn(Q, K, V, key_padding_mask=~pad_mask)

        output = output.squeeze(1)
        output = self.Trans_attn_layernorm(output)

        Z_u = self.Trans_PFFN(output)
        return Z_u


    '''
    DiffGenerator: Bridge-based Diffusion POI Generation model
        Input:  User Personalized Interest Distribution Z_u, 
                User Personalized Transition Representation $ \mathcal{R}_u $ as a context-aware condition embedding
        Return: A sampled future possible user preferences hat_Z_trg
    '''
    def DiffGenerator(self, Z_u, R_u, target=None):
        local_embs = Z_u
        condition_embs = R_u.detach()

        T = torch.full(Z_u.shape, gol.conf['T']).to(gol.device)

        # Reverse-time Generation Process
        hat_Z_trg = self.SDEdiff.ReverseSDE_gener(local_embs, condition_embs, T)

        loss_div = None
        if target is not None: # training phase
            t_sampled = torch.randint(1, self.step_num, Z_u.shape, device=gol.device) / self.step_num

            # get marginal probability
            mean, std = self.SDEdiff.marginal_prob(target, local_embs, t_sampled, T)
            z = torch.randn_like(target)
            perturbed_data = mean + std * z

            # train a time-dependent score-based neural network to estimate marginal probability
            _, score, weights = self.SDEdiff.D_doise(perturbed_data, local_embs, condition_embs, t_sampled, T)

            # Fisher divergence loss_div
            loss_div = (torch.square(score - target) * weights).mean()

        return hat_Z_trg, loss_div


    '''Get Cross-entropy recommendation loss_rec and Fisher divergence loss_div'''
    def getTrainLoss(self, batch):
        usr, pos_lbl, exclude_mask, seqs, G_u, cur_time = batch

        E_P = self.poi_emb
        POI_embs = self.poi_emb
        if gol.conf['dropout']:
            POI_embs = self.dropout(POI_embs)

        R_u = self.POIGraphRep(POI_embs, G_u)
        Z_u = self.InterestGenerator(POI_embs, seqs, R_u)
        hat_Z_trg, loss_div = self.DiffGenerator(Z_u, R_u, target = POI_embs[pos_lbl])
        hat_Y = 10 * torch.matmul(R_u, E_P.t()) + torch.matmul(hat_Z_trg, E_P.t())

        loss_rec = self.CEloss(hat_Y, pos_lbl)
        return loss_rec, loss_div  


    def forward(self, seqs, G_u):
        E_P = self.poi_emb

        R_u = self.POIGraphRep(E_P, G_u)
        Z_u = self.InterestGenerator(E_P, seqs, R_u)
        hat_Z_trg, _ = self.DiffGenerator(Z_u, R_u)

        '''
        E_P    = [#POI, hid_dim] 
        E_P.T  = [hid_dim, #POI] 
        R_u    = [batch_size, hid_dim]
        Z_u    = [batch_size, hid_dim]
        hat_Y  = [batch_size, #POI]
        '''

        hat_Y = 10 * torch.matmul(R_u, E_P.t()) + torch.matmul(hat_Z_trg, E_P.t()) 
        return hat_Y


def Seq_MASK(lengths, max_len=None): 
    lengths_shape = lengths.shape  
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1))).reshape(lengths_shape)