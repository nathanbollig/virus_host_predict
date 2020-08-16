# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:03:50 2020

@author: NBOLLIG
"""
import numpy as np

class Merged2Models():
    """
    Inputs to the constructor is a list of two YoungModel objects.
    """
    def __init__(self, M):
        assert(len(M) == 2)
        self.model_list = M
        assert(len(self.model_list[0].dataset.fs) == len(self.model_list[1].dataset.fs))
        self.feature_num = len(self.model_list[0].dataset.fs)
        self.X_test_pools = self.get_pooled_test_sets()
        self.y_concat, self.source = self.get_concatenated_labels()
        
    """
    Merge the two test sets for each feature set. For now, we will assume the base models have disjoint
    negative classes and so we will not worry about exclusions. We also assume there will be no duplicates in the 
    merged test sets.
    
    Returns:
        X_test_pools - a list of pooled test sets parallel to the models' feature sets
    """
    def get_pooled_test_sets(self):
        X_test_pools = []
        
        ym0 = self.model_list[0]
        ym1 = self.model_list[1]
                
        # Pool test sets
        for i in range(self.feature_num):
            fs0 = ym0.dataset.fs[i]
            fs1 = ym1.dataset.fs[i]
            
            X_tst0 = fs0.X_tst
            X_tst1 = fs1.X_tst
            
            if X_tst0.shape[1] == X_tst1.shape[1]:
                X_test_pool = np.vstack((X_tst0, X_tst1))
            else:
                X_test_pool = 'N/A'
                
            X_test_pools.append(X_test_pool) 
    
        return X_test_pools
    
    """
    Get concatenated list of test labels from model 0 and model 1.
    Returns: 
        y_concat - concatenated list of test labels, kept in binary form wrt original model in whose test set the instance belonged
        source - array of length X_test_pools.shape[0] with 0 if from model 0 and 1 if from model 1
    """
    def get_concatenated_labels(self):
        
        ym1 = self.model_list[0]
        ym2 = self.model_list[1]
        
        mask1 = ym1.dataset.ds['trn/tst']=='test'
        y_test1 = np.asarray(ym1.dataset.ds[mask1]['y'],dtype=int)
       
        mask2 = ym2.dataset.ds['trn/tst']=='test'
        y_test2 = np.asarray(ym2.dataset.ds[mask2]['y'],dtype=int)
        
        s0 = np.full((y_test1.shape[0],), 0, dtype=int) # array of zeros
        s1 = np.full((y_test2.shape[0],), 1, dtype=int) # array of ones
        
        source = np.concatenate((s0, s1)) # 0 if from model 1, 1 if from model 2
        
        return np.concatenate((y_test1, y_test2)), source
    
    """
    Get column of the confusion matrix corresponding to the given instance.
    
    The columns are:
        || model 0 host || model 1 host || neither model host ||
    
    Inputs:
        i - index of instance in pooled test set
    """
    def get_column(self, i):
                   
        source = self.source[i]
        y = self.y_concat[i]
        
        if source==0 and y==1:
            return 0
        elif source==1 and y==1:
            return 1
        else:
            return 2
    
    """
    Get row of the confusion matrix corresponding to the given prediction.
    
    The rows are:
        || model 0 positive ||
        || model 0 negative ||
        || model 1 positive ||
        || model 1 negative ||
    
    Inputs:
        pred - prediction
        m = model
    """
    def get_row(self, pred, m):
        assert(m==0 or m==1)
        
        if m==0 and pred==1:
            return 0
        elif m==0 and pred==0:
            return 1
        elif m==1 and pred==1:
            return 2
        else:
            return 3
        
    """
    Test each model on the pooled test set
    """
    def test_merged(self):
        CM = np.zeros((4,3), dtype=int)
        
        for fs in range(self.feature_num):
            if fs>1:
                X = self.X_test_pools[fs] # loop through test pools defined by all feature sets
                if type(X) != str: # set to string 'N/A' if it was not possible to create pooled test set
                    for i in range(X.shape[0]): # Step through test instances
                        x = X[i].reshape(1, -1) # represents a single instance
                                            
                        for m in range(2):
                            pred = self.model_list[m].models[fs].predict(x).item()
                            row = self.get_row(pred, m)
                            col = self.get_column(i)
                            
                            CM[row][col] += 1
            
        return CM
                        
                        
                    
                
    
        