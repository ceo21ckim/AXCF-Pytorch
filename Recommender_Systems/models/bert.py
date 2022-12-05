import time
import tqdm 
import os 

from __init__ import * 
from models.utils import *

import torch
import torch.nn as nn 

from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, args, dataframe, tokenizer):
        self.tokenizer = tokenizer 
        self.data = dataframe 
        self.reviews = dataframe.text 
        self.labels = dataframe.stars
        self.max_seq_length = args.max_seq_length

        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        review = self.reviews[idx]

        inputs = self.tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=True, 
            max_length=self.max_seq_length, 
            padding='max_length', 
            return_token_type_ids=True, 
            truncation=True
        )

        input_ids = inputs['input_ids']
        masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return (
            torch.tensor(input_ids, dtype=torch.long), # token_ids
            torch.tensor(masks, dtype=torch.long), # attention_mask
            torch.tensor(token_type_ids, dtype=torch.long), # token_type_ids
            torch.tensor(self.labels[idx], dtype = float) # labels
        )


class BERTClassifier(nn.Module):
    def __init__(self, args, bertmodel):
        super(BERTClassifier, self).__init__()
        self.bert = bertmodel
        self.classifier = nn.Linear(args.hidden_dim, 1)
        self.dropout = nn.Dropout(args.dr_rate)

    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        pooler = output['pooler_output']
        pooler = self.dropout(pooler)
        fc_layer = self.classifier(pooler)
        return fc_layer 


def bert_evaluate(args, model, test_loader, criterion):
    loss, acc = 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc='Evaluating...'):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0], 
                      'attention_mask': batch[1], 
                      'token_type_ids': batch[2]}
            label = batch[3]

            pred_y = model(**inputs).squeeze()
            loss = criterion(pred_y, label)

            acc += calc_accuracy(pred_y, label) / len(pred_y)
            loss += loss.item() / len(pred_y)

    acc /= len(test_loader)
    loss /= len(test_loader)

    return acc, loss


def bert_train(args, model, train_loader, test_loader, optimizer, criterion):
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    set_seed(args)
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = 0, 0

        model.train()
        for batch in tqdm.tqdm(train_loader, desc='Traninig...'):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0], 
                      'attention_mask': batch[1], 
                      'token_type_ids': batch[2]}
            label = batch[3]

            pred_y = model(**inputs).sequeeze()
            loss = criterion(pred_y, label)
            
            train_acc += calc_accuracy(pred_y, label) / len(pred_y)
            train_loss += loss.item() / len(pred_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_acc, valid_loss = bert_evaluate(args, model, test_loader, criterion)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)


        end_time = time.time()
        elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)
        print(f'epoch [{epoch}/{args.num_epochs}], elapsed time: {elapsed_mins}m, {elapsed_secs:.2f}s')
        print(f'train loss: {train_loss:.4f}\ttrain accuracy: {train_acc*100:.2f}%')
        print(f'valid loss: {valid_loss:.4f}\ttest accuracy: {valid_acc*100:.2f}%\n')

        if best_loss > valid_loss :
            best_loss = valid_loss 
            if not os.path.exists(args.save_parameters):
                os.makedirs(args.save_parameters)

            torch.save(model.state_dict(), os.path.join(args.save_parameters, f'bert_parameters.pt'))
    
    return {
        'train_loss': train_losses, 
        'train_acc': train_accs, 
        'valid_loss': valid_losses, 
        'valid_acc': valid_accs
    }
