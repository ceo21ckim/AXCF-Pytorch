from settings import * 
from models.metrics import *

from tqdm import tqdm 

import dgl 
import torch 
from torch import nn

from scipy.sparse import coo_matrix 
from scipy.sparse import vstack 
from scipy import sparse 
import numpy as np 

from torch.utils.data import TensorDataset, Dataset, DataLoader


class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats 
        self.out_feats = out_feats 
        self.linear1 = nn.Linear(self.in_feats, self.out_feats)
        self.linear2 = nn.Linear(self.in_feats, self.out_feats)
        
    def forward(self, L, selfLoop, features):
        L1 = L + selfLoop
        L2 = L.cuda()
        L1 = L1.cuda()
        inter_feats = torch.sparse.mm(L2, features)
        inter_feats = torch.mul(inter_feats, features)
        
        inter_part1 = self.linear1(torch.sparse.mm(L1, features))
        inter_part2 = self.linear2(torch.sparse.mm(L2, inter_feats))
        
        return inter_part1 + inter_part2 
    
class NGCF(nn.Module):
    def __init__(self, num_users, num_items, rating, latent_dim, layers):
        super(NGCF, self).__init__()
        self.num_users = num_users 
        self.num_items = num_items 
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.GNNLayers = nn.ModuleList()
        self.L = self.buildLaplacian(rating)
        self.leakyReLU = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.num_users + self.num_items)
        
        self.fc_layer1 = nn.Linear(layers[-1] * (len(layers)) * 2, 64)
        self.fc_layer2 = nn.Linear(64, 32) 
        self.fc_layer3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        for f, t in zip(layers[:-1], layers[1:]):
            self.GNNLayers.append(GNNLayer(f, t))
        
    def getSparseEye(self, num):
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i, val)
        
    def buildLaplacian(self, rt):
        rt_item = rt['business_id'] + self.num_users 
        uiMat = coo_matrix((rt['stars'], (rt['user_id'], rt['business_id'])))
        
        uiMat_upperPart = coo_matrix((rt['stars'], (rt['user_id'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.num_items, self.num_users + self.num_items))
        
        A = vstack([uiMat_upperPart, uiMat])
        selfLoop = sparse.eye(self.num_users, self.num_items)
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row 
        col = L.col 
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data)
        return SparseL 
    
    def getFeatureMat(self):
        uids = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        iids = torch.LongTensor([i for i in range(self.num_items)]).cuda()
        
        user_emb = self.user_embedding(uids)
        item_emb = self.item_embedding(iids)
        features = torch.cat([user_emb, item_emb], dim=0)
        return features 
    
    def forward(self, uids, iids):
        iids = iids + self.num_users 
        uids = list(uids.cpu().data)
        iids = list(iids.cpu().data)
        
        features = self.getFeatureMat()
        final_emb = features.clone()
        
        for gnn in self.GNNLayers:
            features = gnn(self.L, self.selfLoop, features)
            features = self.relu(features)
            final_emb = torch.cat([final_emb, features.clone()], dim=1)

        user_emb = final_emb[uids]
        item_emb = final_emb[iids]
        emb = torch.cat([user_emb, item_emb], dim=1)
        
        emb = self.fc_layer1(emb)
        emb = self.relu(emb)
        
        emb = self.fc_layer2(emb)
        
        emb = self.fc_layer3(emb)
        emb = self.sigmoid(emb)
        output = emb.flatten()
        return output 

class GraphDataset(Dataset):
    def __init__(self, dataframe):
        super(Dataset, self).__init__()
        
        self.uid = list(dataframe['user_id'])
        self.iid = list(dataframe['business_id'])
        self.ratings = list(dataframe['stars'])
    
    def __len__(self):
        return len(self.uid)
    
    def __getitem__(self, idx):
        uid = self.uid[idx]
        iid = self.iid[idx]
        rating = self.ratings[idx]
        
        return (uid, iid, rating)

def get_graph_loader(args, dataset, shuffle, num_workers, gcmc=False):
    if gcmc:
        d_set = GCMCDataset(dataset)
    else:
        d_set = GraphDataset(dataset)

    set_seed(args)
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, rating, latent_dim, layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users 
        self.num_items = num_items
        self.layers = layers
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.GNNLayers = nn.ModuleList()
        self.L = self.buildLaplacian(rating)
        self.leakyReLU = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.num_users + self.num_items)
        
        self.fc_layer1 = nn.Linear(layers[-1] * (len(layers)) * 2, 64)
        self.fc_layer2 = nn.Linear(64, 32)
        self.fc_layer3 = nn.Linear(32, 1)
        self._init_weight()
        
        for f, t in zip(layers[:-1], layers[1:]):
            self.GNNLayers.append(GNNLayer(f, t))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
    def getSparseEye(self, num):
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i, val)
        
    def buildLaplacian(self, rt):
        rt_item = rt['business_id'] + self.num_users 
        uiMat = coo_matrix((rt['stars'], (rt['user_id'], rt['business_id'])))
        
        uiMat_upperPart = coo_matrix((rt['stars'], (rt['user_id'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.num_items, self.num_users + self.num_items))
        
        A = vstack([uiMat_upperPart, uiMat])
        selfLoop = sparse.eye(self.num_users, self.num_items)
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row 
        col = L.col 
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data)
        return SparseL 
    
    def getFeatureMat(self):
        uids = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        iids = torch.LongTensor([i for i in range(self.num_items)]).cuda()
        
        user_emb = self.user_embedding(uids)
        item_emb = self.item_embedding(iids)
        features = torch.cat([user_emb, item_emb], dim=0)
        return features 
    
    def forward(self, uids, iids):
        iids = iids + self.num_users 
        uids = list(uids.cpu().data)
        iids = list(iids.cpu().data)
        
        features = self.getFeatureMat()
        final_emb = features.clone()
        
        for i, gnn in enumerate(self.GNNLayers):
            features = gnn(self.L, self.selfLoop, features)
            if (i + 1) == len(self.layers):
                features = self.leakyReLU(features)
            final_emb = torch.cat([final_emb, features.clone()], dim=1)

        user_emb = final_emb[uids]
        item_emb = final_emb[iids]
        emb = torch.cat([user_emb, item_emb], dim=1)
        
        emb = self.fc_layer1(emb)
        emb = self.leakyReLU(emb)
        
        emb = self.fc_layer2(emb)
        emb = self.leakyReLU(emb)
        
        emb = self.fc_layer3(emb)
        output = emb.flatten()
        return output 


def graph_train(args, model, train_loader, valid_loader, optimizer, criterion):
    best_loss = float('inf')
    train_losses, valid_losses = [], []
    for epoch in range(1, args.num_epochs + 1):
        train_loss = 0.0

        model.train()
        for batch in tqdm(train_loader, desc='training...'):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'uids':   batch[0], 
                      'iids':   batch[1]}
            
            gold_y = batch[2].float()
            

            pred_y = model(**inputs)
            
            loss = criterion(pred_y, gold_y)
            loss = torch.sqrt(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        valid_loss , outputs = graph_evaluate(args, model, valid_loader, criterion)
        valid_losses.append(valid_loss)
        

        print(f'Epoch: [{epoch}/{args.num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}')

        if best_loss > valid_loss:
            best_loss = valid_loss
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{model._get_name()}_parameters.pt'))

    return {
        'train_loss': train_losses, 
        'valid_loss': valid_losses
    }, outputs


def graph_evaluate(args, model, test_loader, criterion):
    output = []
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='evaluating...'):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'uids':   batch[0], 
                      'iids':   batch[1]}
            gold_y = batch[2].float()
            
            pred_y = model(**inputs)
            output.append(pred_y)
            
            loss = criterion(pred_y, gold_y)
            loss = torch.sqrt(loss)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss, output


## GC-MC

def GCMCDataset(dataframe):
    uids = torch.LongTensor(dataframe.loc[:, 'user_id'])
    iids = torch.LongTensor(dataframe.loc[:, 'business_id'])
    ratings = torch.LongTensor(dataframe.loc[:, 'stars'])

    graph = dgl.heterograph({
    ('user', 'preference', 'item'): (uids, iids), 
    ('item', 'preference-by', 'user') : (iids, uids)
    })

    graph.edges['preference'].data['rating'] = ratings 
    graph.edges['preference-by'].data['rating'] = ratings 

    return TensorDataset(uids, iids, ratings)


class MinibatchSampler:
    def __init__(self, graph, num_layers):
        self.graph = graph 
        self.num_layers = num_layers 

    def sample(self, batch):
        users, items, ratings = zip(*batch)
        users = torch.stack(users)
        items = torch.stack(items)
        ratings = torch.stack(ratings)

        pair_graph = dgl.heterograph(
            {('user', 'preference', 'item') : (users, items)}, 
            num_nodes_dict = {'user' : self.graph.number_of_nodes('user'), 'item': self.graph.number_of_nodes('item')}
        )

        pair_graph = dgl.compact_graphs(pair_graph)

        pair_graph.edata['rating'] = ratings

        seeds = {'user': pair_graph.nodes['user'].data[dgl.NID], 
                 'item': pair_graph.nodes['item'].data[dgl.NID]}
        blocks = self.construct_blocks(seeds, (users, items))

        for feature_name in self.graph.nodes['user'].data.keys():
            blocks[0].srcnodes['user'].data[feature_name] = self.graph.nodes['user'].data[feature_name][blocks[0].sronodes['user'].data[dgl.NID]]

        for feature_name in self.graph.nodes['item'].data.keys():
            blocks[0].srcnodes['item'].data[feature_name] = self.graph.nodes['item'].data[feature_name][blocks[0].srcnodes['item'].data[dgl.NID]]

        return pair_graph, blocks

    def construct_blocks(self, seeds, user_item_pairs_to_remove):
        blocks = []
        users, items = user_item_pairs_to_remove
        for i in range(self.num_layers):

            sampled_graph = dgl.in_subgraph(self.graph, seeds)

            sampled_eids = sampled_graph.edges['preference'].data[dgl.EID]
            sampled_eids_rev = sampled_graph.edges['preference-by'].data[dgl.EID]
            
            _, _, edges_to_remove = sampled_graph.edge_ids(users, items, etype='preference', return_uv=True)
            _, _, edges_to_remove_rev = sampled_graph.edge_ids(items, users, etype='preference-by', return_uv=True)
            
            sampled_with_edges_removed = sampled_graph
            if len(edges_to_remove) > 0:
                sampled_with_edges_removed = dgl.remove_edges(
                    sampled_with_edges_removed, edges_to_remove, 'preference')
                sampled_eids = sampled_with_edges_removed.edges['preference'].data[dgl.EID]
            if len(edges_to_remove_rev) > 0:
                sampled_with_edges_removed = dgl.remove_edges(
                    sampled_with_edges_removed, edges_to_remove_rev, 'preference-by')
                sampled_eids_rev = sampled_with_edges_removed.edges['preference-by'].data[dgl.EID]
            
            # Create a block from the sampled graph.
            block = dgl.to_block(sampled_with_edges_removed, seeds)
            blocks.insert(0, block)
            seeds = {'user': block.srcnodes['user'].data[dgl.NID],
                     'item': block.srcnodes['item'].data[dgl.NID]}
            
            # Copy the ratings to the edges of the sampled block
            block.edges['preference'].data['rating'] = \
                self.graph.edges['preference'].data['rating'][sampled_eids]
            block.edges['preference-by'].data['rating'] = \
                self.graph.edges['preference-by'].data['rating'][sampled_eids_rev]
            
        return blocks