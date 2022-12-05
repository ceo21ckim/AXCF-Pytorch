import random
import numpy as np 
import torch 

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