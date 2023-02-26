from tqdm import tqdm 
import wandb

import os 

from settings import *
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

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.num_users = args.num_users
        self.num_items = args.num_items 
        self.latent_dim = args.latent_dim 

        self.user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim)

        self.mlp_layer = nn.Sequential(
            nn.Linear(2*self.latent_dim, self.latent_dim), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim//2, self.latent_dim//4)
        )
        self.fc_layer = nn.Linear(self.latent_dim//4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, uid, iid):
        user_emb = self.user_emb(uid)
        item_emb = self.item_emb(iid)

        inputs = torch.concat([user_emb, item_emb], axis=1)
        outputs = self.mlp_layer(inputs)
        outputs = self.fc_layer(outputs)
        outputs = self.sigmoid(outputs)
        return outputs 

class NeuMF(nn.Module):
    def __init__(self, args):
        super(NeuMF, self).__init__()

        self.num_users = args.num_users 
        self.num_items = args.num_items
        self.latent_dim = args.latent_dim

        self.mf_user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.mf_item_emb = nn.Embedding(self.num_items, self.latent_dim)

        self.mlp_user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.mlp_item_emb = nn.Embedding(self.num_items, self.latent_dim)

        self.mlp_layer = nn.Sequential(
            nn.Linear(2*self.latent_dim, self.latent_dim), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim//2, self.latent_dim//4)
        )
        self.fc_layer = nn.Linear(self.latent_dim//4 + self.latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, uid, iid):
        mf_user_emb = self.mf_user_emb(uid)
        mf_item_emb = self.mf_item_emb(iid)

        mlp_user_emb = self.mlp_user_emb(uid)
        mlp_item_emb = self.mlp_item_emb(iid)

        mf_outputs = torch.mul(mf_user_emb, mf_item_emb)

        mlp_inputs = torch.concat([mlp_user_emb, mlp_item_emb], axis = 1)
        mlp_outputs = self.mlp_layer(mlp_inputs)

        outputs = torch.concat([mlp_outputs, mf_outputs], axis=1)
        outputs = self.fc_layer(outputs)
        outputs = self.sigmoid(outputs)
        return outputs 


class MFDataset(Dataset):
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

def get_mf_loader(args, dataset, shuffle, num_workers):
    d_set = MFDataset(dataset)
    set_seed(args)
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)


def mf_train(args, model, train_loader, test_loader, optimizer, criterion):
    train_losses = []
    valid_losses = []
    best_loss = float('inf')

    set_seed(args)
    model.train()
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        train_loss, valid_loss = 0.0, 0.0

        for batch in train_loader:
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

        valid_loss, _ = mf_evaluate(args, model, test_loader, criterion)
        valid_losses.append(valid_loss)

        wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'epoch': epoch, 'lr':args.learning_rate})

        if epoch % 5 == 0:
            print(f'Epoch: [{epoch}/{args.num_epochs}]')
            print(f'Train Loss: {train_loss:.5f}')
            print(f'Valid Loss: {valid_loss:.5f}')

        if best_loss > valid_loss:
            best_loss = valid_loss
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{model._get_name()}_parameters.pt'))
    wandb.finish()

    return {
        'train_loss': train_losses, 
        'valid_loss': valid_losses
    }


def mf_evaluate(args, model, test_loader, criterion):
    losses = 0
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(b.to(args.device) for b in batch)

            inputs = {'uid':    batch[0], 
                      'iid':    batch[1]}
            gold_y = batch[2]

            pred_y = model(**inputs).squeeze()
            preds.append(pred_y)
                
            loss = criterion(pred_y, gold_y)
        
            losses += loss.item()
        losses /= len(test_loader)
    preds = torch.concat(preds)
    return losses, preds
