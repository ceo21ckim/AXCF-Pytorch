from settings import * 
from tqdm import tqdm 
import wandb 

import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader

class SADataset(Dataset):
    def __init__(self, args, ratings):
        self.uids = ratings['user_id']
        self.iids = ratings['business_id']
        self.reviews = ratings['text']
        self.ratings = ratings['stars']
        self.tokenizer = args.tokenizer
        self.max_seq_length = args.max_length

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        iid = self.iids[idx]
        rating = self.ratings[idx]
        review = self.reviews[idx]
        review = self.tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=False, 
            max_length=self.max_seq_length, 
            padding='max_length', 
            return_token_type_ids=False, 
            truncation=True, 
            return_attention_mask = False
        )['input_ids']

        return (
            torch.tensor(uid, dtype=torch.long), 
            torch.tensor(iid, dtype=torch.long), 
            torch.tensor(review, dtype=torch.long), 
            torch.tensor(rating, dtype=torch.float)
        )

def get_seq_loader(args, d_set, shuffle, num_workers):
    data_set = SADataset(args, d_set)
    return DataLoader(data_set, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

class RNNClassifier(nn.Module):
    def __init__(self, args):
        super(RNNClassifier, self).__init__()
        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.latent_dim = args.latent_dim 
        self.num_layers = args.num_layers
        self.vocab_size = args.vocab_size
        self.dr_rate = args.dr_rate
        self.bidirectional = args.bidirectional
        self.max_length = args.max_length
        self.dropout = nn.Dropout(self.dr_rate)

        self.user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim)
        self.review_emb = nn.Embedding(self.vocab_size, self.latent_dim*2)

        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim//2)
        )

        self.lstm = nn.RNN(
            input_size = self.latent_dim*2, 
            hidden_size = self.latent_dim, 
            num_layers = self.num_layers, 
            batch_first=True, 
            dropout = self.dr_rate, 
            bidirectional = self.bidirectional
        )
        self.fc_layer = nn.Linear(self.latent_dim*2 + self.latent_dim//2 if self.bidirectional else self.latent_dim + self.latent_dim//2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, uids, iids, reviews):
        user_emb = self.user_emb(uids)
        item_emb = self.item_emb(iids)

        if self.dr_rate:
            self.dropout(reviews)

        review_emb = self.review_emb(reviews)

        o, h = self.lstm(review_emb)

        if self.bidirectional:
            lstm_outs = torch.concat([h[-1], h[-2]], dim=-1)
        else:
            lstm_outs = h[-1]

        mf_inputs = torch.concat([user_emb, item_emb], dim=-1)
        mf_outs = self.interaction_mlp(mf_inputs)
        
        outs = torch.concat([mf_outs, lstm_outs], dim=1)
        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
        return outs        



class LSTMClassifier(nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()
        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.latent_dim = args.latent_dim 
        self.num_layers = args.num_layers
        self.vocab_size = args.vocab_size
        self.dr_rate = args.dr_rate
        self.bidirectional = args.bidirectional
        self.max_length = args.max_length
        self.dropout = nn.Dropout(self.dr_rate)

        self.user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim)
        self.review_emb = nn.Embedding(self.vocab_size, self.latent_dim*2)

        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim//2)
        )

        self.lstm = nn.LSTM(
            input_size = self.latent_dim*2, 
            hidden_size = self.latent_dim, 
            num_layers = self.num_layers, 
            batch_first=True, 
            dropout = self.dr_rate, 
            bidirectional = self.bidirectional
        )
        self.fc_layer = nn.Linear(self.latent_dim*2 + self.latent_dim//2 if self.bidirectional else self.latent_dim + self.latent_dim//2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, uids, iids, reviews):
        user_emb = self.user_emb(uids)
        item_emb = self.item_emb(iids)

        if self.dr_rate:
            self.dropout(reviews)

        review_emb = self.review_emb(reviews)

        o, (h, c) = self.lstm(review_emb)

        if self.bidirectional:
            lstm_outs = torch.concat([h[-1], h[-2]], dim=-1)
        else:
            lstm_outs = h[-1]

        mf_inputs = torch.concat([user_emb, item_emb], dim=-1)
        mf_outs = self.interaction_mlp(mf_inputs)
        
        outs = torch.concat([mf_outs, lstm_outs], dim=1)
        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
        return outs          


def evaluate(args, model, test_loader, criterion):
    outs = []
    loss = 0 
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = tuple(b.to(args.device) for b in batch)

            inputs = {
                'uids':     batch[0],
                'iids':     batch[1], 
                'reviews':  batch[2]
            }

            gold_y = batch[3]

            pred_y = model(**inputs).squeeze()

            loss += criterion(pred_y, gold_y)
            outs.append(pred_y.detach().cpu())
        loss /= len(test_loader)
    return loss, outs 

def train(args, model, train_loader, valid_loader, optimizer, criterion):
    train_losses, valid_losses = [], []
    best_loss = float('inf')
    for epoch in range(1, args.num_epochs+1):
        train_loss = 0 
        model.train()
        for batch in tqdm(train_loader):
            batch = tuple(b.to(args.device) for b in batch)

            inputs = {
                'uids':     batch[0],
                'iids':     batch[1], 
                'reviews':  batch[2]
            }

            gold_y = batch[3]

            pred_y = model(**inputs).squeeze()
            loss = criterion(pred_y, gold_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss 
        train_loss /= len(train_loader)
        valid_loss, _ = evaluate(args, model, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch: [{epoch}/{args.num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}')

        wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'epoch': epoch, 'lr':args.learning_rate})

        if best_loss > valid_loss:
            best_loss = valid_loss 
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{model._get_name()}_parameters.pt'))
    wandb.finish()
    return train_losses, valid_losses 