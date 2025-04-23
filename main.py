import torch
import gol
import numpy as np
from torch.utils.data import DataLoader
from pprint import pformat

from model import Model
from dataset import GraphData, collate_eval, getDatasets, collate_edge, ACC_at_k, MRR


def eval_model(model: Model, eval_set: GraphData):
    Ks = [1, 5, 10, 20]
    result = {'ACC': np.zeros(len(Ks)), 'MRR': 0.}
    eval_loader = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_eval)

    with torch.no_grad():
        model.eval()
        tot_cnt = 0
        for idx, batch in enumerate(eval_loader):
            u, pos_list, exclude_mask, seqs, seq_graph, cur_time = batch
            item_score = model(seqs, seq_graph)
            seq_graph.mean_interv[seq_graph.mean_interv.isnan()] = seq_graph.mean_interv[~seq_graph.mean_interv.isnan()].mean()

            item_score[exclude_mask] = -1e10
            item_score = item_score.cpu()

            for score, label in zip(item_score, pos_list):
                ranked_idx = np.argsort(-score)[: max(Ks)]
                rank_results = label[ranked_idx]

                ACC = []
                for K in Ks:
                    ACC.append(ACC_at_k(rank_results, K, 1))
                mrr = MRR(rank_results)

                result['ACC'] += ACC             
                result['MRR'] += mrr
                tot_cnt += 1

    result['ACC'] /= tot_cnt 
    result['MRR'] /= tot_cnt
    return result

def train_eval(model: Model, datasets):
    trn_set, val_set, tst_set = datasets
    trn_loader = DataLoader(trn_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_edge)
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])
    batch_num = len(trn_set) // gol.BATCH_SZ
    best_val_epoch, best_val_ACC = 0., 0.
    ave_tot, ave_rec, ave_fis = 0., 0., 0.
    tst_result = None

    for epoch in range(gol.EPOCH):
        model.train()
        for idx, batch in enumerate(trn_loader):
            loss_rec, loss_div  = model.getTrainLoss(batch)
            tot_loss = loss_rec + gol.conf['zeta'] * loss_div

            opt.zero_grad()
            tot_loss.backward()
            opt.step()
            if idx % (batch_num // 2) == 0:
                gol.pLog(f'Batch {idx} / {batch_num}, Total_loss: {tot_loss.item():.5f}' + f' = loss_rec: {loss_rec.item():.5f} + loss_div: {loss_div.item():.5f}')

            ave_tot += tot_loss.item()
            ave_rec += loss_rec.item()
            ave_fis += loss_div.item()

        ave_tot /= batch_num
        ave_rec /= batch_num
        ave_fis /= batch_num

        val_results = eval_model(model, val_set)

        gol.pLog(f'Avg Epoch {epoch} / {gol.EPOCH}, Total_Loss: {ave_tot:.5f}' + f' = loss_rec: {ave_rec:.5f} + loss_div: {ave_fis:.5f}')
        gol.pLog(f'ACC@5: {val_results["ACC"][1]:.5f}')
        if epoch - best_val_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without valid improvement.')
            break

        if val_results["ACC"][1] > best_val_ACC or epoch == 0:
            best_val_epoch, best_val_ACC = epoch, val_results["ACC"][1]
            # test phase: 
            tst_result = eval_model(model, tst_set)
            gol.pLog(f'New test top@k result at k = {1, 5, 10, 20}: \n {pformat(tst_result)}')

        gol.pLog(f'Best valid ACC@5 at epoch {best_val_epoch}\n')

    return tst_result, best_val_epoch

if __name__ == '__main__':

    n_user, n_poi, datasets = getDatasets(gol.DATA_PATH, gol.dataset)
    POI_model = Model(n_user, n_poi)

    POI_model = POI_model.to(gol.device)
    
    gol.pLog(f'Dropout probability: {gol.conf["dp"] if gol.conf["dropout"] else 0}')
    num_params = 0
    for param in POI_model.parameters():
        num_params += param.numel()
    gol.pLog(f'The Number of Parameters of Model is {num_params}')
    gol.pLog(f'-------------------Start Training---------------------\n')

    test_result, best_epoch = train_eval(POI_model, datasets)
    gol.pLog(f'\n Training on {gol.dataset.upper()} Finished, Best Valid at epoch {best_epoch}')
    gol.pLog(f'**Best Test Top@k Result at k = {1, 5, 10, 20} is**\n{pformat(test_result)}')

