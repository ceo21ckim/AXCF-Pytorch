import torch 

def calc_accuracy(pred_y, true_y):
    pred_y = torch.sigmoid(pred_y)
    return ((pred_y > 0.5) == true_y).sum().detach().cpu().item()

def sentiment_score(x):
    if x >= 3.5 : return 1
    elif x < 3.5 : return 0