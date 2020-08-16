# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:50:56 2020

@author: NBOLLIG
"""

from dataset import DataSet

"""
The YoungModel class will point to an existing DataSet and include additional fields for downstream needs.
"""

class YoungModel():
    
    def __init__(self, dataset):
        self.models = [] # List of classifier objects parallel to the fs list (one model for each feature set)
        self.dataset = dataset # pointer to dataset object