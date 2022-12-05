import time
import tqdm 

import torch
import torch.nn as nn 
from __init__ import *
from settings import * 
from models.utils import * 

from torch.utils.data import Dataset, DataLoader

class LSTMDataset(Dataset):
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
            add_special_tokens=False,
            max_length=self.max_seq_length, 
            padding='max_length', 
            return_token_type_ids=False, 
            return_attention_mask=False,
            truncation=True
        )

        input_ids = inputs['input_ids']

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype = float) # labels
        )


class LSTMClassifier(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, bidirectional=False, drop_rate=None):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim ,
            hidden_size = hidden_dim, 
            num_layers = n_layers, 
            batch_first = True, 
            bidirectional=bidirectional, 
            dropout = drop_rate
        )
        self.drop_rate = drop_rate

        self.fc = nn.Linear(2*hidden_dim if bidirectional else hidden_dim, n_classes)

        self.dropout = nn.Dropout(drop_rate)
        self._init_weight()
    
    def forward(self, idx):
        embedding = self.embedding(idx)
        if self.drop_rate != 0:
            self.dropout(embedding)
        
        output, (hidden, _) = self.lstm(embedding)

        if self.lstm.bidirectional:
            output = torch.cat([hidden[-1], hidden[-2]], dim = -1)
            output = self.dropout(output)
        else:
            output = self.dropout(output)
        
        output = self.fc(output)
        return output
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

def get_dataloader(args, d_set, tokenizer):
    set_seed(args)
    dataset = LSTMDataset(args, d_set, tokenizer)
    return DataLoader(dataset, batch_size=args.batch_size)


def lstm_evaluate(args, model, test_loader, criterion):
    valid_acc, valid_loss = 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc='Evaluating..'):
            batch = tuple(t.to(args.device) for t in batch)
            reviews, labels = batch 

            pred_y = model(reviews).sequeeze()
            loss = criterion(pred_y, labels)

            valid_acc += calc_accuracy(pred_y, labels) / len(pred_y)
            valid_loss += loss.item() / len(pred_y)
    
    valid_acc /= len(test_loader)
    valid_loss /= len(test_loader)
    
    return valid_acc, valid_loss 


def lstm_train(args, model, train_loader, valid_loader, optimizer, criterion):
    best_loss = float('inf')

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = 0, 0
        valid_loss, valid_acc = 0, 0

        start_time = time.time()

        model.train()
        for batch in tqdm.tqdm(train_loader, desc='Training..'):
            batch = tuple(t.to(args.device) for t in batch)
            reviews, labels = batch 

            pred_y = model(reviews).squeeze()
            loss = criterion(pred_y, labels)

            train_acc += calc_accuracy(pred_y, labels) / len(pred_y)
            train_loss += loss.item() / len(pred_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        valid_acc, valid_loss = lstm_evaluate(args, model, valid_loader, criterion)

        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)

        end_time = time.time()
        elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)
        print(f'epoch [{epoch}/{args.num_epochs}] | elapsed time: {elapsed_mins}m, {elapsed_secs:.2f}s')
        print(f'train loss: {train_loss:.6f}\ttrain accuracy: {train_acc*100:.2f}%')
        print(f'valid loss: {valid_loss:.6f}\tvalid accuracy: {valid_acc*100:.2f}% \n')


        if best_loss > valid_loss :
            best_loss = valid_loss 
            if not os.path.exists(args.save_parameters):
                os.makedirs(args.save_parameters)

            torch.save(model.state_dict(), os.path.join(args.save_parameters, f'lstm_parameters.pt'))

    return {
        'train_loss': train_losses, 
        'train_acc': train_accs, 
        'valid_loss': valid_losses, 
        'valid_acc': valid_accs
    }
