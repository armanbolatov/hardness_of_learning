import argparse
import sympy
import numpy as np
import random
import pandas as pd
from fractions import Fraction
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os


parser = argparse.ArgumentParser(description='p',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--p', type=int, default=11)
args = parser.parse_args()
p = args.p

#get gradients
def get_grads(el_a):
    X = []
    for el in [int((el_a*el)%p) for el in orig_X]:
        bin_el = bin(el)[2:]
        bin_el = "0"*(bit-len(bin_el))+bin_el
        X.append([int(el) for el in bin_el])
    X = np.array(X, dtype='float32')
    X = torch.from_numpy(X)
    X = X.cuda()

    model.zero_grad()
    logits, probs = model(X)
    loss = criterion(logits, Y)
    loss.backward()
    grads = torch.cat([
            param.grad.detach().flatten()
            for param in model.parameters()  
            if param.grad is not None
        ])
    return grads
    

class BaseModel1(nn.Module):
    def __init__(self, max_len ):
        super(BaseModel1, self).__init__()
        self.linear1 = nn.Linear(max_len, 1000) 
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1)
        
    def forward(self, inputs):
        h1 = torch.sigmoid(self.linear1(inputs))
        h2 = torch.sigmoid(self.linear2(h1))
        logits = self.linear3(h2)
        probs = torch.sigmoid(logits)
        return logits, probs



bit = len(bin(p)[2:])

a = list(range(1,p))
df = pd.DataFrame({'p':p, 'a':a})

#data for calculation of v(w)
orig_X = [Fraction(l) for l in range(1,p)]
orig_Y = np.array([int(bin(int(el))[-1]) for el in orig_X]).reshape(-1, 1)
Y = torch.from_numpy(orig_Y)
Y = Y.float().cuda()

criterion = nn.BCEWithLogitsLoss()

#data for calculation of g(w)
zs = []
for el in range(1,p):
    bin_el = bin(el)[2:]
    bin_el = "0"*(bit-len(bin_el))+bin_el
    zs.append([int(el) for el in bin_el])
zs = np.array(zs, dtype='float32')
zs = torch.from_numpy(zs)

#number of models
models_num=20
results = []
h_grads = []
#train
for i in range(models_num):
    model = BaseModel1(bit)
    model.cuda()
    all_grads = []
    #calculate v(w)
    for a in df['a'].values:
        a_grads = get_grads(a).detach().cpu().numpy()
        all_grads.append(a_grads)

    all_grads = np.vstack(all_grads)
    norms = np.linalg.norm(all_grads - all_grads.mean(axis=0), axis=1)**2
    results += [[i, a, norm]  for a, norm in zip(df['a'].values, norms)]

    #calculate g(w)
    for z in zs.cuda():
        model.zero_grad()
        logits, probs = model(z)
        probs.backward()
        grads = torch.cat([
                param.grad.detach().flatten()
                for param in model.parameters()  
                if param.grad is not None
            ])
        h_grads.append([i, (torch.norm(grads)**2).item()])

results_df = pd.DataFrame(results, columns=['w_i','a', 'norm'])
h_df = pd.DataFrame(h_grads, columns=['w_i','h'])
results_df = results_df.groupby('w_i').agg({'norm':['mean','std']})
results_df.columns = [col[0]+"_"+col[1] for col in results_df.columns]
results_df = results_df.reset_index()

h_df = h_df.groupby('w_i').agg({'h':['mean','std']})
h_df.columns = [col[0]+"_"+col[1] for col in h_df.columns]
h_df = h_df.reset_index()
results_df = results_df.merge(h_df, how='left', on='w_i')
results_df['scaled_norm'] = results_df['norm_mean']/results_df['h_mean']
results_df['bit'] = bit
results_df['p']=p
results_df.to_csv(f'results/{p}.csv', index=False)
