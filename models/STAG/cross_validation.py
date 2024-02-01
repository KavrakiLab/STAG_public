import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import os
import copy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

from .encode_structure import *
from .model import *
from .utils import *

def get_graphs(data_table,data_dir):
    df = pd.read_csv(data_table,index_col=0)
    graphs = []
    # for i,row in tqdm(df.iterrows(), total=df.shape[0]):
    for i,row in tqdm(df.iterrows(), total=df.shape[0]):
        graph = get_graph(data_dir+'/'+str(int(i))+'.pdb',row.label,i)
        add_edge_data(graph)
        graph.index = i
        graphs.append(graph)
    return graphs

def run(loader,model,criterion,optimizer,training=False):
    if training:
        model.train()
    else:
        model.eval()
    loss = 0
    y_truth,y_hat = [],[]
    for data in loader:
        data = data.to('cuda')
        out = model(data.x,data.edge_index,data.edge_attr,data.batch)
        if training:
            loss = criterion(out, data.y.float().to('cuda'))
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)
#             print(grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss += loss.item()

        y_truth = np.concatenate((y_truth,data.y.detach().to('cpu').numpy()))
        out = out.detach().to('cpu').numpy()
        if out.ndim == 0:
            out = [out]
        y_hat = np.concatenate((y_hat,out))
    return roc_auc_score(y_truth,y_hat),loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def cross_validation(data_table,data_dir,k=5,verbose=False):
    print('encoding structures as graphs')
    graphs = get_graphs(data_table,data_dir)
    df = pd.read_csv(data_table,index_col=0)
    df['fold'] =  np.random.randint(0,k,size=df.shape[0])
    AUCs = []
    print('begining '+str(k)+' fold cross validation')
    for i in range(k):
        data = copy.deepcopy(graphs)
        train_graphs = [graph for graph in data if df.loc[graph.fold]['fold'] not in [i]]
        test_graphs = [graph for graph in data if df.loc[graph.fold]['fold'] == i]
        y_test_ids = np.array([graph.index for graph in test_graphs])
        train_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=graph.label) for graph in train_graphs]
        test_data_list = [torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=graph.label) for graph in test_graphs]
        train_loader = torch_geometric.loader.DataLoader(train_data_list,batch_size=256,shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(test_data_list,batch_size=64,shuffle=False)

        stag = STAG().to('cuda')
        stag.apply(init_weights)
        optimizer = torch.optim.Adam(stag.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-np.mean(df['label']))/np.mean(df['label'])))

        best_test_auc = 0
        stop = 0

        with tqdm(range(350)) as t:
            for epoch in t:
                stag.train()
                train_auc,loss = run(train_loader,stag,criterion,optimizer,True)
                with torch.no_grad():
                    stag.eval()
                    test_auc,_ = run(test_loader,stag,criterion,optimizer)

                if test_auc > best_test_auc:
                    stop = 0
                    best_test_auc = test_auc
                    torch.save(stag.state_dict(), 'output/STAG_model.pt')

                    with torch.no_grad():
                        stag.eval()
                        y_test_targets,pred = [],[]
                        for data in test_loader:
                            data = data.to('cuda')
                            out = stag(data.x,data.edge_index,data.edge_attr,data.batch)
                            y_test_targets = np.concatenate((y_test_targets,data.y.detach().to('cpu').numpy()))
                            out = out.detach().to('cpu').numpy()
                            if out.ndim == 0:
                                out = [out]
                            pred = np.concatenate((pred,out))

                        final_preds = pd.DataFrame(columns=['index','label','pred'])
                        for j in range(len(pred)):
                            final_preds = final_preds.append({'index':y_test_ids[j],'label':y_test_targets[j],'pred':pred[j]},ignore_index=True)

                t.set_description(f"test_auc: {str(best_test_auc)}")
                stop = stop + 1
                if stop > 25 and epoch > 50:
                    if verbose:
                        print('early stopping')
                    break

            if verbose:
                print('best epoch:',epoch-stop+1)
                print('test_AUC:',best_test_auc)

            AUCs.append(best_test_auc)
            # if best_test_auc == np.max(AUCs):
            #     torch.save(stag.state_dict(), 'output/STAG_model.pt')
            # final_preds.to_csv('repeated_cross_fold_sample/STAG_'+peptide+'_'+str(seed)+'_'+str(i)+'.csv')

            del data
            del train_graphs
            del test_graphs
            del train_data_list
            del test_data_list
            del train_loader
            del test_loader
            del stag
            del optimizer
            del criterion
            torch.cuda.empty_cache()

    print('#'*50)
    print('ROC-AUCs: ',AUCs)
    print('saving best model')

    return AUCs
