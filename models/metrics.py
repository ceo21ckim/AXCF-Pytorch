import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import torch 


def accuracy(pred_y, true_y):
    pred_y = torch.sigmoid(pred_y)
    return ((pred_y > 0.5) == true_y).sum().detach().cpu().item()

def sentiment_score(x):
    if x >= 3.5 : return 1
    elif x < 3.5 : return 0

def dcg(label, k):
    label = np.asfarray(label)[:k]
    if label.size:
        return label[0] + np.sum(label[1:] / np.log2(np.arange(2, label.size + 1)))

    return 0

def ndcg(dataframe, k):
    ndcg_list = []
    for uid in dataframe.user_id.unique():
        label_temp = dataframe.loc[dataframe.user_id == uid]['stars'].tolist()

        idcg = dcg(sorted(label_temp, reverse=True), k)

        if not idcg:
            return 0 

        ndcg_list.append(dcg(label_temp, k) / idcg)
    return np.mean(ndcg_list)


# def cf_metrics(args, dataframe, model, top_k:list):
#     # metrics for Collaborative Filtering
#     item = dataframe.groupby(['user_id'])['stars'].sum()
#     precision_k, recall_k, f1_k, ndcg_k = [], [], [], []
#     for k in top_k:
#         precision, recall, f1_score, ndcg_score = [], [], [], []
#         for uid in tqdm(dataframe.loc[:, 'user_id'].unique(), desc=f'{model._get_name()} Top@{k} evaluating...'):
#             new_df = dataframe.loc[dataframe.loc[:, 'user_id']==uid].copy()
#             uids = torch.tensor(new_df.user_id.values).to(args.device)
#             iids = torch.tensor(new_df.business_id.values).to(args.device)
            
#             model.eval()
#             with torch.no_grad():
#                 yhat = model(uids, iids).squeeze()
#             yhat = (yhat > 0.5).float().cpu().numpy()
#             new_df.loc[:, 'yhat'] = yhat 
#             new_df = new_df.sort_values(by = ['yhat'], ascending=False).head(k)

#             pr_temp = sum(new_df.loc[:, 'stars']) / k 
#             re_temp = sum(new_df.loc[:, 'stars']) / item[uid] if item[uid] != 0 else 0 
#             pr_re = pr_temp + re_temp 
#             f1_temp = (2 * pr_temp * re_temp) / pr_re if pr_re != 0 else 0
#             precision.append(pr_temp)
#             recall.append(re_temp)
#             f1_score.append(f1_temp)
#             ndcg_score.append(ndcg(new_df, k))
        
#         precision_k.append(np.mean(precision))
#         recall_k.append(np.mean(recall))
#         f1_k.append(np.mean(f1_score))
#         ndcg_k.append(np.mean(ndcg_score))

#     outputs = pd.DataFrame({
#         'recall': recall_k, 
#         'precision': precision_k, 
#         'f1_score': f1_k, 
#         'ndcg': ndcg_k
#     })
#     outputs.index = [top_k]
#     return outputs 


def metrics(dataframe, top_k):
    # metrics for Sentiment Analysis
    item = dataframe.groupby(['user_id'])['stars'].sum()
    precision_k, recall_k, f1_k, ndcg_k = [], [], [], []
    for k in top_k:
        precision, recall, f1_score, ndcg_score = [], [], [], []
        for uid in tqdm(dataframe.loc[:, 'user_id'].unique(), desc=f'Tok@{k} evaluating..'):
            new_df = dataframe.loc[dataframe.loc[:, 'user_id'] == uid].copy()

            new_df = new_df.sort_values(by = ['yhat'], ascending=False).head(k)
            pr_temp = sum(new_df.loc[:, 'stars']) / k 
            re_temp = sum(new_df.loc[:, 'stars']) / item[uid] if item[uid] != 0 else 0 
            pr_re = pr_temp + re_temp 
            f1_temp = (2 * pr_temp * re_temp) / pr_re if pr_re != 0 else 0
            precision.append(pr_temp)
            recall.append(re_temp)
            f1_score.append(f1_temp)
            ndcg_score.append(ndcg(new_df, k))
        
        precision_k.append(np.mean(precision))
        recall_k.append(np.mean(recall))
        f1_k.append(np.mean(f1_score))
        ndcg_k.append(np.mean(ndcg_score))

    outputs = pd.DataFrame({
        'recall': recall_k, 
        'precision': precision_k, 
        'f1_score': f1_k, 
        'ndcg': ndcg_k
    }, index=top_k)
    return outputs 


def proposed_metrics(dataframe, top_k):
    # metrics for Proposed Models 
    item = dataframe.groupby(['user_id'])['stars'].sum()
    precision_k, recall_k, f1_k, ndcg_k = [], [], [], []
    for k in top_k :
        precision, recall, f1_score, ndcg_score = [], [], [], []
        for uid in tqdm(dataframe.loc[:, 'user_id'].unique(), desc='evaluating...'):
            user_opinion = dataframe.loc[dataframe.loc[:, 'user_id'] == uid, 'user_tag'].values[0]
            new_df = dataframe.loc[dataframe.loc[:, 'user_id'] == uid].copy()

            new_df = new_df.sort_values(by = ['yhat'], ascending=False).head(k)                       
            new1 = new_df.loc[(new_df.loc[:, 'user_tag'] == user_opinion) & (new_df.loc[:, 'rest_tag'] == user_opinion)]
            new2 = new_df.loc[(new_df.loc[:, 'user_tag'] == user_opinion) & (new_df.loc[:, 'rest_tag'] != user_opinion)]

            new_df = pd.concat([new1, new2], axis=0)

            
            pr_temp = sum(new_df.loc[:, 'stars']) / k 
            re_temp = sum(new_df.loc[:, 'stars']) / item[uid] if item[uid] != 0 else 0 
            pr_re = pr_temp + re_temp 
            f1_temp = (2 * pr_temp * re_temp) / pr_re if pr_re != 0 else 0
            precision.append(pr_temp)
            recall.append(re_temp)
            f1_score.append(f1_temp)
            ndcg_score.append(ndcg(new_df, k))
        
        precision_k.append(np.mean(precision))
        recall_k.append(np.mean(recall))
        f1_k.append(np.mean(f1_score))
        ndcg_k.append(np.mean(ndcg_score))

    outputs = pd.DataFrame({
        'recall': recall_k, 
        'precision': precision_k, 
        'f1_score': f1_k, 
        'ndcg': ndcg_k
    }, index=top_k)

    return outputs 