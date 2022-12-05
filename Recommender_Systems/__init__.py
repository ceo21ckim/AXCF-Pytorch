import os, random
import numpy as np 
import torch 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SEM_DIR = os.path.join(BASE_DIR, 'dataset/SemEval')
YELP_DIR = os.path.join(BASE_DIR, 'dataset/Yelp2018')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figure')
PARAM_DIR = os.path.join(BASE_DIR, 'bert-san-finetune')

SAVE_PATH = os.path.join(BASE_DIR, 'baseline_parameters')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = elapsed_time - elapsed_mins*60 
    return elapsed_mins, elapsed_secs