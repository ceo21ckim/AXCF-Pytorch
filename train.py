import pandas as pd 
import argparse
import wandb 
from torch import optim, nn 

from settings import * 

from models.metrics import sentiment_score
from models.matrix_factorization import GMF, NeuMF, MLP, get_mf_loader, mf_train
from models.sequential import get_seq_loader, RNNClassifier, LSTMClassifier
import models.sequential as seq 
from transformers import DistilBertTokenizer 

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
parser.add_argument('--latent_dim', default=64, type=int)
parser.add_argument('-epoch', '--num_epochs', default=100, type=int)
parser.add_argument('-batch', '--batch_size', default=512, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--num_users', default=25_369, type=int)
parser.add_argument('--num_items', default=44_553, type=int)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--criterion', type=str, required=True)
parser.add_argument('--optimizer', type=str, required=True)
parser.add_argument('--max_length', default=256, type=int)
parser.add_argument('-layers', '--num_layers', default=2, type=int)
parser.add_argument('--bidirectional', default=True, type=bool)
parser.add_argument('--dr_rate', default=0, type=float)

args = parser.parse_args()

wandb.init(
    project='my-Thesis', 
    config = {
    'learning_rate':args.learning_rate, 
    'epochs': args.num_epochs, 
    'name': args.model,
    'optimizer': args.optimizer, 
    'batch_size': args.batch_size, 
    'latent_dim': args.latent_dim, 
    'display_name': args.model 
    }
)



def train(model, dataset, optimizer='SGD', criterion='BCE'):
    args.tokenizer = tokenizer
    args.vocab_size = tokenizer.vocab_size
    d_train, d_valid = dataset 
    # get_dataset
    if model in ['GMF', 'NeuMF', 'MLP']:
        train_loader = get_mf_loader(args, d_train, shuffle=True, num_workers= 4 if args.device == 'cuda' else 1)
        valid_loader = get_mf_loader(args, d_valid, shuffle=True, num_workers= 4 if args.device == 'cuda' else 1)
        trainer = mf_train
    
    elif model in ['RNN', 'LSTM']:
        train_loader = get_seq_loader(args, d_train, shuffle=True, num_workers= 4 if args.device == 'cuda' else 1)
        valid_loader = get_seq_loader(args, d_valid, shuffle=True, num_workers= 4 if args.device == 'cuda' else 1)
        trainer = seq.train

    elif model in ['AXCF', 'NGCF', 'GC-MC']:
        pass

    if model == 'GMF':
        model = GMF(args).to(args.device)

    elif model == 'NeuMF':
        model = NeuMF(args).to(args.device)

    elif model == 'MLP':
        model = MLP(args).to(args.device)

    elif model == 'LSTM':
        model = LSTMClassifier(args).to(args.device)

    elif model == 'RNN':
        model = RNNClassifier(args).to(args.device)

    



    # get optimizer 
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-4)
    
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=1e-4)

    
    if criterion == 'BCE':
        criterion = nn.BCELoss().to(args.device)
    
    elif criterion == 'MSE':
        criterion = nn.MSELoss().to(args.device)

    elif criterion == 'MAE':
        criterion = nn.L1Loss().to(args.device)

    
    train_loss, valid_loss = trainer(args, model, train_loader, valid_loader, optimizer, criterion)

    return train_loss, valid_loss 

if __name__ == '__main__':

    train_path = os.path.join(YELP_DIR, 'train.csv')
    valid_path = os.path.join(YELP_DIR, 'valid.csv')

    d_train = pd.read_csv(train_path, encoding='utf-8-sig')
    d_valid = pd.read_csv(valid_path, encoding='utf-8-sig')

    d_train.loc[:, 'stars'] = d_train.loc[:, 'stars'].apply(sentiment_score)
    d_valid.loc[:, 'stars'] = d_valid.loc[:, 'stars'].apply(sentiment_score)

    train_loss, valid_loss = train(args.model, [d_train, d_valid])