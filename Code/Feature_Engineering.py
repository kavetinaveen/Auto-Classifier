#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from patsy import dmatrices
from Correlations_Visualizations import correlations


class transformations_interactions(object):
    def __init__(self, df):
        self.df = df
        cor = correlations(df)
        _, _, self.categorical_cols, self.numerical_cols =  cor.data_stats()
        
    def log(self, x):
        if x not in self.numerical_cols:
            warnings.warn('Object contains character values, please pass numerical variables only for log transformation')
            return
        else:
            if sum(self.df[x] < 0) > 0:
                warnings.warn("Negative values introduced NaN")
            return np.log(self.df[x])
    
    def power(self, x, k):
        if x not in self.numerical_cols:
            warnings.warn('Object contains character values, please pass numerical variables only for power transformation')
            return
        return self.df[x]**k
    
    def sin(self, x):
        if x not in self.numerical_cols:
            warnings.warn('Object contains character values, please pass numerical variables only for sin transformation')
            return
        return np.sin(self.df[x])
    
    def cos(self, x):
        if x not in self.numerical_cols:
            warnings.warn('Object contains character values, please pass numerical variables only for cos transformation')
            return
        return np.cos(self.df[x])
    
    def exp(self, x):
        if x not in self.numerical_cols:
            warnings.warn('Object contains character values, please pass numerical variables only for exp transformation')
            return
        return np.exp(self.df[x])
    
    def one_hot_encoding(self, x):
        return pd.get_dummies(self.df[x])
    
    def interaction(self, x1, x2, y):
        df = self.df
        print(x1)
        print(x2)
        if (x1 in self.categorical_cols) & (x2 in self.categorical_cols):
            y, X = dmatrices(y + ' ~ ' + 'C(' + x1 + ')' + '*' + 'C(' + x2 + ')', df, return_type = "dataframe")
            y = np.ravel(y)
        elif (x1 in self.categorical_cols) & (x2 in self.numerical_cols):
            y, X = dmatrices(y + ' ~ ' + 'C(' + x1 + ')' + '*' + x2, df, return_type = "dataframe")
            y = np.ravel(y)
        elif (x1 in self.numerical_cols) & (x2 in self.categorical_cols):
            y, X = dmatrices(y + ' ~ ' + x1 + '*' + 'C(' + x2 + ')', df, return_type = "dataframe")
            y = np.ravel(y)
        elif (x1 in self.numerical_cols) & (x2 in self.numerical_cols):
            y, X = dmatrices(y + ' ~ ' + x1 + '*' + x2, df, return_type = "dataframe")
            y = np.ravel(y)
        return X
        
    # Process function will create all transformations for all numerical columns and one-hot vectorization for categorical columns
    def process(self, y, k = 2, transformations = None, interactions = None):
        df = self.df
        if transformations is not None:
            for x in self.numerical_cols:
                if 'ln' in transformations:
                    df[x+'_ln'] = self.log(x)
                    self.numerical_cols = self.numerical_cols + [x+'_ln']
                if 'power' in transformations:
                    df[x+'_power'] = self.power(x, k)
                    self.numerical_cols = self.numerical_cols + [x+'_power']
                if 'sin' in transformations:
                    df[x+'_sin'] = self.sin(x)
                    self.numerical_cols = self.numerical_cols + [x+'_sin']
                if 'cos' in transformations:
                    df[x+'_cos'] = self.cos(x)
                    self.numerical_cols = self.numerical_cols + [x+'_cos']
                if 'exp' in transformations:
                    df[x+'_exp'] = self.exp(x)
                    self.numerical_cols = self.numerical_cols + [x+'_exp']
#                 self.numerical_cols = self.numerical_cols+[x+'_ln', x+'_power', x+'_sin', x+'_cos', x+'_exp']
        features = [x for x in df.columns if x != y]
        if interactions is not None:
            if interactions == 'all':
                for feature_A in features:
                    for feature_B in features:
                        if feature_A > feature_B:
                            X = self.interaction(feature_A, feature_B, y)
                            X = X[[x for x in X.columns if ':' in x]]
                            df = pd.concat([df.reset_index(drop = True), X], axis = 1)
            else:
                for i in range(len(interactions)):
                    X = self.interaction(interactions[i][0], interactions[i][1], y)
                    X = X[[x for x in X.columns if ':' in x]]
                    df = pd.concat([df.reset_index(drop = True), X], axis = 1)
        
        cat_ivs = [x for x in self.categorical_cols if x != y]
        df = pd.concat([df[[x for x in df.columns if not x in cat_ivs]].reset_index(drop = True), \
                        self.one_hot_encoding(cat_ivs)], axis = 1)
        
        return df

