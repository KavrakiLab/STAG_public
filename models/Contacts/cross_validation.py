from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from .encode_structure import *
from .model import *
from .utils import *
import pickle

def get_contact_data(data_table,data_dir):
    df = pd.read_csv(data_table,index_col=0)
    contact_names = ['contacts_'+str(i) for i in range(400)]
    df[contact_names] = [None]*400
    for i,row in tqdm(df.iterrows(), total=df.shape[0]):
        df.loc[i,contact_names] = get_contacts(data_dir+'/'+str(i)+'.pdb')
    return df

def cross_validation(data_table,data_dir,k=5,verbose=False):
    print('encoding TCR-pMHC contacts')
    df = get_contact_data(data_table,data_dir)
    contact_names = ['contacts_'+str(i) for i in range(400)]
    df['fold'] =  np.random.randint(0,k,size=df.shape[0])
    bests = []

    print('begining '+str(k)+' fold cross validation')
    for i in range(k):
        train_data = df[df['fold'] != i]
        test_data = df[df['fold'] == i]

        y_train_targets = np.array(train_data.label)
        y_test_targets = np.array(test_data.label)
        y_test_ids = np.array(test_data.index)

        x_train = np.array(train_data[contact_names])
        x_test = np.array(test_data[contact_names])

        clf = RandomForestClassifier(n_estimators=500,max_features=0.75,class_weight='balanced_subsample',max_samples=0.5)
        clf.fit(x_train,y_train_targets)

        pred = clf.predict_proba(x_test)[:,1]
        auc = roc_auc_score(y_test_targets,pred)
        bests.append(auc)

        final_preds = pd.DataFrame(columns=['index','label','pred'])
        for j in range(len(y_test_ids)):
            final_preds = final_preds.append({'index':y_test_ids[j],'label':y_test_targets[j],'pred':pred[j]},ignore_index=True)
        # final_preds.to_csv('repeated_cross_fold_sample/contacts_RF_'+peptide+'_'+str(seed)+'_'+str(i)+'.csv')

        if verbose:
            print('test_AUC: ',auc)
    if verbose:
        print('MEAN AUC: ',np.mean(bests))

    print('#'*50)
    print('ROC-AUCs: ',bests)
    print('saving best model')
    with open('output/contacts_model.pkl','wb') as f:
        pickle.dump(clf,f)
    return bests
