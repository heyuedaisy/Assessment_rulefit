#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:48:38 2020

@author: nanetsu
"""

import numpy as np
import pandas as pd
from rulefit import RuleFit

boston_data=pd.read_csv('boston.csv',index_col=0)
#boston_data
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',None)
boston_data

y=boston_data.medv.values
X=boston_data.drop('medv',axis=1)
features=X.columns
X=X.values

from sklearn.model_selection import train_test_split
#from sklearn,metrics import roc_curve,roc_auc_score, accuracy_score
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
 
from sklearn.ensemble import GradientBoostingRegressor
rgb=RuleFit(Cs=None,cv=3,exp_rand_tree_size=True,lin_standardise=True,
            lin_trim_quantile=0.025,max_rules=2000,memory_par=0.01,model_type='rl',
            random_state=None,rfmode='regress',sample_fract='default',
            tree_generator=GradientBoostingRegressor(alpha=0.9,
                                                 criterion='friedman_mse',
                                                 init=None, learning_rate=0.02,
                                                 loss='ls', max_depth=100,
                                                 max_features=None,
                                                 max_leaf_nodes=15,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators=500,
                                                 n_iter_no_change=None,
                                                 presort='auto',
                                                 random_state=572,
                                                 subsample=0.46436099318265595,
                                                 tol=0.0001,
                                                 validation_fraction=0.1,
                                                 verbose=0, warm_start=False),
                                                     tree_size=3)
rgb.fit(x_train,y_train)
y_pred=rgb.predict(x_test)
rules=rgb.get_rules()



def scaled_absolute_error(y_test,y_pred):
    e1=np.mean(y_test-y_pred)
    e2=np.mean(y_test-np.median(y_test))
    return np.round(e1/e2,4)

scaled_absolute_error(y_test,y_pred)
