"""Threshold-0.5 classification metrics and reliability bins.

MorphoNup reports raw probabilities (mean of the forward and reverse-complement views); no post-hoc calibration.
"""
import numpy as np
from sklearn import metrics as sm


def reliability(y,p,bins=10):
    import pandas as pd
    edges=np.linspace(0,1,bins+1);rows=[]
    for i in range(bins):
        take=(p>=edges[i])&((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        rows.append(dict(bin=i,lower=edges[i],upper=edges[i+1],count=int(take.sum()),
                         predicted=float(p[take].mean()) if take.any() else np.nan,
                         observed=float(y[take].mean()) if take.any() else np.nan))
    return pd.DataFrame(rows)


def scores(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities.')
    pred=p>=.5;tn,fp,fn,tp=map(int,sm.confusion_matrix(y,pred,labels=[0,1]).ravel())
    rel=reliability(y,p);ece=float((rel['count']*(rel.predicted-rel.observed).abs()).sum()/len(y))
    return dict(n=len(y),accuracy=sm.accuracy_score(y,pred),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        precision=sm.precision_score(y,pred,zero_division=0),f1=sm.f1_score(y,pred,zero_division=0),mcc=sm.matthews_corrcoef(y,pred),
        balanced_accuracy=sm.balanced_accuracy_score(y,pred),kappa=sm.cohen_kappa_score(y,pred),roc_auc=sm.roc_auc_score(y,p),
        pr_auc=sm.average_precision_score(y,p),nll=sm.log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),
        brier=sm.brier_score_loss(y,p),ece=ece,tn=tn,fp=fp,fn=fn,tp=tp)
