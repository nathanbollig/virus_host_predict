# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:27:10 2020

@author: NBOLLIG

Derived from run_TrainTest.ipynb from Young paper.
"""

import os
import pandas as pd
import pickle
import random
import numpy as np
import csv
import timeit
from collections import defaultdict, Counter
from pathlib import Path
from Bio import SeqIO
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score
import sys
lib_dir = '../mylibs'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)   
    
import vhdb as vhdb
from featureset import FeatureSet
from dataset import DataSet
from young_model import YoungModel
from merged_models import Merged2Models

def get_confusion( y_test,y_pred):
    FN=0
    FP=0
    TP=0
    TN=0
    #missed = 0
    for y,yp in zip(y_test,y_pred):
        #print (y_true,yp,yp2,yprob)
        if y == True and yp == False:
            FN += 1
        elif y == False and yp == True:
            FP +=1
        elif y == True and yp == True:
            TP +=1
        else:
            TN +=1
    spec = round(TN/(TN+FP),2) if (TN + TP) else 'NA'
    sens = round(TP/(TP+FN),2) if (TP+FN) else 'NA'
    fdr  = round(FP/(TP+FP),2) if (TP+FP) else 'NA'
    
    #NB 8-6-2020 Need to change 'prec' in the below to 'Prec' otherwise causes error in results2CSV
    res  = { 'Acc':round((TP+TN)/(len(y_test)),2),'Spec':spec, 'Sens':sens,\
           'Prec':fdr,'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    return res


def test_prediction(fs,y_trn,y_tst):
    clf = make_pipeline(StandardScaler(),SVC(kernel='linear',probability=True) )
    clf.fit(fs.X_trn, y_trn)
    y_pred = clf.predict(fs.X_tst)
    y_pred_probs= clf.predict_proba(fs.X_tst)[:,1]
    AUC =round( roc_auc_score(y_tst, y_pred_probs),3)  
    confusion = get_confusion(y_tst,y_pred)
    confusion.update({'AUC':AUC})
    return confusion, clf


def main(): 
    # Input file  
    subsetfile = '../inputs/Euk_all.csv'
    label_info = pd.read_csv(subsetfile)
    subsets = label_info.apply(tuple, axis=1).tolist()
    print(f'{len(subsets)} datasets : {subsets[0]}')
    for s in subsets:
        print(s)
    vhdbfile = '../inputs/VHDB_25_1_2019.p'
    with open(vhdbfile, 'rb') as f:
        V_H = pickle.load( f)
    hosts = V_H.hosts
    viruses = V_H.viruses
    print (f'{len(viruses)} viruses and {len(hosts)} hosts')
    
    # Output file for the results
    results_file = f'../results/{Path(subsetfile).stem}_results.csv '
    print (f'Results will be saved in: {results_file}')
    
    # Get feature_sets list
    features = ['DNA','AA','PC','Domains']
    kmer_lists = [[1,2,3,4,5,6,7,8,9], # dna 
                  [1,2,3,4],           # aa
                  [1,2,3,4,5,6] ,      #pc
                  [0]]  
    feature_sets = [f'{f}_{k}' for i,f in enumerate(features) for k in kmer_lists[i] ]

    # SPECIFY WHICH SUBSETS TO USE
    new_subsets = [subsets[24], subsets[16]]

    # Run experiment
    all_results =[]
    M = [] # Collect a list of YoungModel objects
    for subset in new_subsets:
        print  (subset)
        data = DataSet(subset,V_H,feature_sets=feature_sets)
        (label,label_tax,pool,pool_tax,baltimore) = subset
        print  (label,label_tax,pool,pool_tax,baltimore)
        print((data.ds.groupby(['y','trn/tst']).count(),'\n'))
        mask = data.ds['trn/tst']=='train'
        y_train = np.asarray(data.ds[mask]['y'],dtype=int)
        y_test = np.asarray(data.ds[~mask
                                   ]['y'],dtype=int)

        # Create a YoungModel object for the DataSet
        ym = YoungModel(data)
        
        for fs in data.fs:            
            # Train models
            results, clf = test_prediction(fs,y_train,y_test)
            results.update ({'N': len(data.ds), 'features':fs.feature, 'k':fs.k})
            print(results)
            data.results2CSV (results,subset, results_file)
            all_results.append(results)
            
            # Update ym object
            ym.models.append(clf)
        
        results_df = pd.DataFrame(all_results)
        M.append(ym)

    # Pooled test set experiment (A4)
    merged_model = Merged2Models(M)
    merged_model.test_merged()

if __name__ == '__main__':
    main()