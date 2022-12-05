from tqdm import tqdm 

import os 
from __init__ import * 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 

class GMF(nn.Module):
    def __init__(self, args):
        super(GMF, self).__init__()

        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.latent_dim = args.latent_dim 

        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)

        self.fc_layer = nn.Linear(args.latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, uid, iid):
        user_emb = self.user_embedding(uid)
        item_emb = self.item_embedding(iid)

        multiply_layer = torch.mul(user_emb, item_emb)
        predict = self.fc_layer(multiply_layer)
        output = self.sigmoid(predict)
        return output 

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.zero_()


class CFDataset(Dataset):
    def __init__(self, dataframe):
        self.user_id = dataframe.user_id
        self.item_id = dataframe.business_id
        self.labels = dataframe.stars 

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        uid = self.user_id[idx]
        iid = self.item_id[idx]
        label = self.labels[idx]

        return (
            torch.tensor(uid, dtype=torch.long), 
            torch.tensor(iid, dtype=torch.long), 
            torch.tensor(label, dtype=torch.float)
        )

def get_dataloader(args, dataset, num_workers):
    d_set = CFDataset(dataset)
    set_seed(args)
    return DataLoader(d_set, batch_size=args.batch_size, num_workers=num_workers)


def train(args, model, train_loader, test_loader, optimizer, criterion):
    train_losses = []
    valid_losses = []
    best_loss = float('inf')

    set_seed(args)
    model.train()
    for epoch in range(1, args.num_epochs + 1):
        train_loss, valid_loss = 0.0, 0.0

        for batch in tqdm(train_loader, desc='Training...'):
            batch = tuple(b.to(args.device) for b in batch)

            inputs = {'uid':    batch[0], 
                      'iid':    batch[1]}
            gold_y = batch[2]

            pred_y = model(**inputs).squeeze()
            loss = criterion(pred_y, gold_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        train_losses.append(train_loss)

        valid_loss = evaluate(args, model, test_loader, criterion, mode=False)
        valid_losses.append(valid_loss)


        print(f'Epoch: [{epoch}/{args.num_epochs}]')
        print(f'Train Loss: {train_loss:.5f}')
        print(f'Valid Loss: {valid_loss:.5f}')

        if best_loss > valid_loss:
            best_loss = valid_loss 
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{model._get_name()}_parameters.pt'))
    
    return {
        'train_loss': train_losses, 
        'valid_loss': valid_losses
    }


def evaluate(args, model, test_loader, criterion, mode=True):
    losses = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating...'):
            batch = tuple(b.to(args.device) for b in batch)

            inputs = {'uid':    batch[0], 
                      'iid':    batch[1]}
            gold_y = batch[2]

            pred_y = model(**inputs).squeeze()
            if mode:
                pred_y = 4 * pred_y + 1
                
            loss = criterion(pred_y, gold_y)
        
            losses += loss.item()
        losses /= len(test_loader)
    return losses
