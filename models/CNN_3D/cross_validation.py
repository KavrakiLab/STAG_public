import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

from .encode_structure import *
from .model import *
from .utils import *

def get_vox(data_table,data_dir):
    df = pd.read_csv(data_table,index_col=0)
    data = []
    for i,row in tqdm(df.iterrows(), total=df.shape[0]):
        vox = torch.from_numpy(voxelize(data_dir+'/'+str(i)+'.pdb')).float()
        data.append((vox,torch.tensor(row.label).float(),i))
    return data

class dataset(Dataset):
    def __init__(self,items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self,i):
        x = self.items[i][0]
        return x,self.items[i][1]

def cross_validation(data_table,data_dir,k=5,verbose=False):
    print('encoding structures as voxels')
    data = get_vox(data_table,data_dir)
    df = pd.read_csv(data_table,index_col=0)
    df['fold'] =  np.random.randint(0,k,size=df.shape[0])
    AUCs = []
    print('begining '+str(k)+' fold cross validation')
    for i in range(k):
        train_data = dataset([d for d in data if df.loc[d[2]]['fold'] not in [i]])
        test_data = dataset([d for d in data if df.loc[d[2]]['fold'] == i])
        y_test_ids = df[df['fold'] == i].index
        train_loader = DataLoader(train_data,batch_size=128,shuffle=True,drop_last=True)
        test_loader = DataLoader(test_data,batch_size=24,shuffle=False,drop_last=True)

        model = Model_3DCNN().float().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-np.mean(df['label']))/np.mean(df['label'])))

        best_test_auc = 0
        stop = 0

        with tqdm(range(50)) as t:
            for epoch in t:
                model.train()
                loss = 0
                stop += 1
                for x,y in train_loader:
                    x = x.to('cuda')
                    x = model(x)
                    loss = criterion(x.squeeze(), y.float().to('cuda'))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                y_test_targets,preds = [],[]
                with torch.no_grad():
                    model.eval()
                    for x,y in test_loader:
                        x = x.to('cuda')
                        out = model(x).squeeze()
                        preds = np.concatenate((preds,out.detach().to('cpu').numpy()))
                        y_test_targets = np.concatenate((y_test_targets,y.detach().to('cpu').numpy()))

                auc = roc_auc_score(y_test_targets,preds)
                t.set_description(f"test_auc: {str(best_test_auc)}")
                if auc > best_test_auc:
                    best_test_auc = auc
                    stop = 0

                    final_preds = pd.DataFrame(columns=['index','label','pred'])
                    for j in range(len(preds)):
                        final_preds = final_preds.append({'index':y_test_ids[j],'label':y_test_targets[j],'pred':preds[j]},ignore_index=True)


                if stop > 5:
                    if verbose:
                        print('early stopping')
                    break
        if verbose:
            print('best_test_AUC: ',best_test_auc)
        AUCs.append(best_test_auc)
        if best_test_auc == np.max(AUCs):
            torch.save(model.state_dict(), 'output/CNN_3D_model.pt')
        # final_preds.to_csv('repeated_cross_fold/3Dcnn_'+peptide+'_'+str(seed)+'_'+str(i)+'.csv')

    print('#'*50)
    print('ROC-AUCs: ',AUCs)
    print('saving best model')

    return AUCs
