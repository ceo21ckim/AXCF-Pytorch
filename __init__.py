import os, random
import numpy as np 
import torch 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SEM_DIR = os.path.join(BASE_DIR, 'dataset/SemEval')
YELP_DIR = os.path.join(BASE_DIR, 'datset/Yelp2018')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figure')
PARAM_DIR = os.path.join(BASE_DIR, 'bert-san-finetune')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)