import numpy as np 

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


# top_k = 5

# item = test.groupby(['user_id'])["stars"].sum()
# precision_k, recall_k, f1_k, ndcg_k = [], [], [], []
# for k in range(5, top_k + 1, 5):
#     precision, recall, f1_score, ndcg_score = [], [], [], []
#     for uid in tqdm(test.loc[:, 'user_id'].unique(), desc='evaluating..'):
#         new_df = test.loc[test.loc[:, 'user_id'] == uid].copy()
#         uids = torch.tensor(new_df.user_id.values).to(device)
#         iids = torch.tensor(new_df.business_id.values).to(device)
#         label = torch.tensor(new_df.stars.values).to(device)
#         model.eval()
#         with torch.no_grad():
#             yhat = model(uids, iids).squeeze()
#         yhat = (yhat > 0.5).float().cpu().numpy()
#         new_df.loc[:, 'yhat'] = yhat
#         new_df = new_df.sort_values(by = ['yhat'], ascending=False).head(k)
        
#         pr_temp = sum(new_df.loc[:, 'stars']) / k 
#         re_temp = sum(new_df.loc[:, 'stars']) / item[uid] if item[uid] != 0 else 0
#         pr_re = pr_temp + re_temp
#         f1_temp = ( 2 * pr_temp * re_temp) / pr_re if pr_re != 0 else 0
#         precision.append(pr_temp)
#         recall.append(re_temp)
#         f1_score.append(f1_temp)
#         ndcg_score.append(ndcg(new_df, k))
#     precision_k.append(np.mean(precision))
#     recall_k.append(np.mean(recall))
#     f1_k.append(np.mean(f1_score))
#     ndcg_k.append(np.mean(ndcg_score))

# results = pd.DataFrame({
#     'precision': precision_k, 
#     'recall': recall_k, 
#     'f1_score': f1_k, 
#     'ndcg': ndcg_k
# }, index=[5])