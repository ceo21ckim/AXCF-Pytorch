
import os 
import pandas as pd 

from models.parsers import bert_args
from models.bert import BERTDataset, BERTClassifier, bert_train
from __init__ import *

from torch import nn, optim 
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer


if __name__ == '__main__':
    args = bert_args

    train_path = os.path.join(YELP_DIR, 'train.csv')
    valid_path = os.path.join(YELP_DIR, 'valid.csv')

    train = pd.read_csv(train_path, encoding='utf-8-sig')
    valid = pd.read_csv(valid_path, encoding='utf-8-sig')

    bert_model = BertModel.from_pretrained(args.bert_name_or_path)
    bert_model = BERTClassifier(args, bert_model).to(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name_or_path)

    trainset = BERTDataset(args, train, tokenizer)
    validset = BERTDataset(args, valid, tokenizer)

    train_loader = DataLoader(trainset, batch_size=args.batch_size)
    valid_loader = DataLoader(validset, batch_size=args.batch_size)

    optimizer = optim.AdamW(bert_model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    results = bert_train(args, bert_model, train_loader, valid_loader, optimizer, criterion)

    save_path = os.path.join(BASE_DIR, 'baseline')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results.to_csv(os.path.join(save_path, 'bert_results.csv'), encoding='utf-8-sig', index=False) 