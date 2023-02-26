import pandas as pd
import argparse
from torch import nn 

from settings import * 

from models.metrics import sentiment_score, metrics 
from models.matrix_factorization import GMF, NeuMF, MLP, get_mf_loader, mf_evaluate
from models.sequential import get_seq_loader, RNNClassifier, LSTMClassifier
import models.sequential as seq 
from transformers import DistilBertTokenizer 

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--num_users', default=25_369, type=int)
parser.add_argument('--num_items', default=44_553, type=int)
parser.add_argument('--latent_dim', default=64, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--k', default=[2, 4, 5, 6, 8, 10, 20], type=list)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--max_length', default=256, type=int)
parser.add_argument('-layers', '--num_layers', default=2, type=int)
parser.add_argument('--bidirectional', default=True, type=bool)
parser.add_argument('--dr_rate', default=0, type=float)
parser.add_argument('--seed', default=42, type=int)

args = parser.parse_args()


def evaluate(model, dataset, criterion='BCE'):
    args.tokenizer = tokenizer
    args.vocab_size = tokenizer.vocab_size

    # get_dataset
    if model in ['GMF', 'NeuMF', 'MLP']:
        test_loader = get_mf_loader(args, dataset, shuffle=False, num_workers= 4 if args.device == 'cuda' else 1)
        evaluater = mf_evaluate
    
    elif model in ['RNN', 'LSTM']:
        test_loader = get_seq_loader(args, dataset, shuffle=False, num_workers= 4 if args.device == 'cuda' else 1)
        evaluater = seq.evaluate

    elif model in ['AXCF', 'NGCF', 'GC-MC']:
        pass

    if model == 'GMF':
        model = GMF(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'GMF_parameters.pt')))

    elif model == 'NeuMF':
        model = NeuMF(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'NeuMF_parameters.pt')))

    elif model == 'MLP':
        model = MLP(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'MLP_parameters.pt')))

    elif model == 'LSTM':
        model = LSTMClassifier(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'LSTMClassifier_parameters.pt')))

    elif model == 'RNN':
        model = RNNClassifier(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'RNNClassifier_parameters.pt')))
    
    if criterion == 'BCE':
        criterion = nn.BCELoss().to(args.device)
    
    elif criterion == 'MSE':
        criterion = nn.MSELoss().to(args.device)

    elif criterion == 'MAE':
        criterion = nn.L1Loss().to(args.device)

    
    test_loss, outputs = evaluater(args, model, test_loader, criterion)

    return test_loss, outputs

if __name__ == '__main__':
    d_test = pd.read_csv(os.path.join(YELP_DIR, 'test.csv'), encoding='utf-8-sig')
    d_test.stars = d_test.stars.apply(sentiment_score)


    test_loss, outputs = evaluate(args.model, d_test)

    if args.model in ['GMF', 'NeuMF', 'MLP']:
        yhat = outputs.detach().cpu()

        d_test.loc[:, 'yhat'] = yhat
    
    elif args.model in ['RNN', 'LSTM']:
        d_test.loc[:, 'yhat'] = torch.concat(outputs, dim=0)

    results = metrics(d_test, args.k)

    csv_path = os.path.join(BASE_DIR, 'results')
    print(results)
    results.to_csv(os.path.join(csv_path, f'{args.model}.csv'), encoding='utf-8-sig')