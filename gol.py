import torch, os, logging, random
import numpy as np
from config import parse_args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pLog(s: str): logging.info(s)

CORES = 16
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# *****Attention: Please store datasets in your path: './data/processed'
DATA_PATH = '../UB-Diff/data/processed'

ARG = parse_args()
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"
if ARG.log is not None: logging.basicConfig(filename=ARG.log, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
else: logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

SEED = ARG.seed
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
EPOCH = ARG.epoch
dataset = ARG.dataset
patience = ARG.patience

seed_torch(SEED)

dist_mat = torch.from_numpy(np.load(os.path.join(DATA_PATH, dataset.upper(), 'dist_mat.npy')))
device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
conf = {'lr': ARG.lr,
        'decay': ARG.decay,
        'num_layer': ARG.layer,
        'num_heads': ARG.num_heads,
        'hidden': ARG.hidden,
        'dropout': ARG.dropout, 'dp': ARG.dp,
        'max_len': ARG.length,
        'interval': ARG.interval,
        'zeta': ARG.zeta,
        'T': ARG.diffsize,
        'beta_min': ARG.beta_min,
        'beta_max': ARG.beta_max,
        'dt': ARG.stepsize
        }