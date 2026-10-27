import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F

def eval_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def eval_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def eval_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def eval_acc(gts, preds):

    accs = []
    for i in range(gts.shape[0]):

        gts_sign = gts[i] > 0
        preds_sign = preds[i] > 0

        total = len(gts[i])
        num_true = (gts_sign == preds_sign).sum()

        if total > 0:
            acc = num_true / total
        else:
            acc = 0

        accs.append(acc)

    return np.asanyarray(accs)

def eval_long_acc(gts, preds, confidence_threshold = 0):
    res = gts[preds > confidence_threshold]
    total = len(res)
    num_true = len(res[res > 0])
    if total > 0:
        acc = num_true / total
    else:
        acc = 0

    return acc, num_true

def eval_short_acc(gts, preds, confidence_threshold = 0):
    res = gts[preds < confidence_threshold]
    total = len(res)
    num_true = len(res[res < 0])
    if total > 0:
        acc = num_true / total
    else:
        acc = 0

    return acc, num_true

def eval_ic(gts, preds):
    ics = []
    for i in range(gts.shape[0]):
        ics.append(stats.pearsonr(gts[i], preds[i])[0])
    return np.asanyarray(ics)

def eval_rank_ic(gts, preds):
    rank_ics = []
    for i in range(gts.shape[0]):
        rank_ics.append(stats.spearmanr(gts[i], preds[i])[0])
    return np.asanyarray(rank_ics)

def eval_long_prec(gts, preds, K=10):
    long_precs = []
    for i in range(gts.shape[0]):
        pred = preds[i]
        gt = gts[i]
        top_k_largest = np.argpartition(pred, -K)[-K:]
        top_return = gt[top_k_largest] * pred[top_k_largest]
        prec = len(top_return[top_return > 0]) / len(top_return)
        long_precs.append(prec)
    return np.asanyarray(long_precs)

def eval_short_prec(gts, preds, K=10):
    short_precs = []
    for i in range(gts.shape[0]):
        pred = preds[i]
        gt = gts[i]
        top_k_smallest = np.argpartition(pred, K)[:K]
        top_return = gt[top_k_smallest] * pred[top_k_smallest]
        prec = len(top_return[top_return > 0]) / len(top_return)
        short_precs.append(prec)
    return np.asanyarray(short_precs)


def eval_all_metrics(gts, preds):
    dict_metrics = {}
    mae = eval_mae(gts, preds)
    rmse = eval_rmse(gts, preds)
    # long_acc = eval_long_acc(gts, preds, -np.mean(gts, axis=0))[0]
    # short_acc = eval_short_acc(gts, preds, np.mean(gts, axis=0))[0]
    ics = eval_ic(gts, preds)
    rank_ics = eval_rank_ic(gts, preds)
    long_k_precs = eval_long_prec(gts, preds)
    short_k_precs = eval_short_prec(gts, preds)
    acc = eval_acc(gts, preds)

    dict_metrics["mae"] = mae
    dict_metrics["rmse"] = rmse
    # dict_metrics["long_acc"] = long_acc
    # dict_metrics["short_acc"] = short_acc
    dict_metrics["acc"] = np.mean(acc)
    dict_metrics["ic"] = np.mean(ics)
    dict_metrics["icir"] = np.mean(ics) / np.std(ics)
    dict_metrics["rankic"] = np.mean(rank_ics)
    dict_metrics["rankicir"] = np.mean(rank_ics) / np.std(rank_ics)
    dict_metrics["long_k_prec"] = np.mean(long_k_precs)
    dict_metrics["short_k_prec"] = np.mean(short_k_precs)
        
    return dict_metrics


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        
        self.mse = nn.MSELoss()
        
    def forward(self, true, pred):

        rmse = torch.sqrt(self.mse(true, pred))

        return rmse

