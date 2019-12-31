#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Loading required packages
import scipy.stats as ss
from collections import Counter
import math 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from Correlations_Visualizations import correlations
from Feature_Engineering import transformations_interactions


# In[79]:


class feature_selection(object):
    
    def __init__(self, df, y, num_feats, transformations, interactions, info_features):
        self.df = df
        self.y = y
        self.num_feats = num_feats
        self.info_features = info_features
        self.info_features_df = df[info_features]
        df = df.drop(info_features, axis = 1)
        cor = correlations(df)
        feat_engineering = transformations_interactions(df)
        _, _, self.categorical_cols, self.numerical_cols =  cor.data_stats()
        self.df_processed = feat_engineering.process(y, transformations = transformations, interactions = interactions)
    
    def cor_selector(self):
        X = self.df_processed
        Y = self.df[self.y]
        print(Y)
        cor_list = []
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], Y)[0, 1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-self.num_feats:]].columns.tolist()
        cor_support = [True if i in cor_feature else False for i in X.columns]
        return cor_support, cor_feature
    
    def chi2_selector(self):
        X = self.df_processed
        Y = self.df[self.y]
        X_norm = MinMaxScaler().fit_transform(X)
        chi_selector = SelectKBest(chi2, k = self.num_feats)
        chi_selector.fit(X_norm, Y)
        chi_support = chi_selector.get_support()
        chi_feature = X.loc[:,chi_support].columns.tolist()
        return chi_support, chi_feature
    
    def recursive_feature_elimination(self):
        X = self.df_processed
        Y = self.df[self.y]
        X_norm = MinMaxScaler().fit_transform(X)
        rfe_selector = RFE(estimator = LogisticRegression(), n_features_to_select = self.num_feats, step = 10, verbose = 5)
        rfe_selector.fit(X_norm, Y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = X.loc[:,rfe_support].columns.tolist()
        return rfe_support, rfe_feature
    
    def select_from_model_lr(self):
        X = self.df_processed
        Y = self.df[self.y]
        X_norm = MinMaxScaler().fit_transform(X)
        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty = "l1"), '1.25*median')
        embeded_lr_selector.fit(X_norm, Y)
        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
        return embeded_lr_support, embeded_lr_feature
    
    def select_from_model_rf(self):
        X = self.df_processed
        Y = self.df[self.y]
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
        embeded_rf_selector.fit(X, Y)
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
        return embeded_rf_support, embeded_rf_feature
    
    '''
    def select_from_model_lgbm():
        X = self.df_processed.drop(y, axis = 1)
        Y = self.df_processed[y]
        lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

        embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
        embeded_lgb_selector.fit(X, y)
        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
        return embeded_lgb_support, embeded_lgb_feature
    '''
    
    def process(self):
        X = self.df_processed
#         cor_support, cor_feature = self.cor_selector()
        chi_support, chi_feature = self.chi2_selector()
        rfe_support, rfe_feature = self.recursive_feature_elimination()
        embeded_lr_support, embeded_lr_feature = self.select_from_model_lr()
        embeded_rf_support, embeded_rf_feature = self.select_from_model_rf()
        
#         'Pearson':cor_support
        feature_selection_df = pd.DataFrame({'Feature':X.columns, 'Chi-2':chi_support,                                              'RFE':rfe_support, 'Logistics':embeded_lr_support,                                     'Random Forest':embeded_rf_support})
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        feature_selection_df = feature_selection_df.sort_values(['Total','Feature'], ascending = False)
        feature_selection_df.index = range(1, len(feature_selection_df)+1)

        output = X[feature_selection_df['Feature'].values[:self.num_feats]]
        output[self.y] = self.df[self.y]
        output[self.info_features] = self.info_features_df

        return output


# In[80]:


#data = pd.read_csv("Bank Marketing Data/bank_marketing.csv", skiprows = 1, names = ['age','job','marital',                                'education','default','balance','housing','loan','contact','day','month',                                'duration','campaign','pdays','previous','poutcome','Class'])


# In[81]:


#feat_selector = feature_selection(data, 'Class', 50)


# In[82]:


#feat_selector.process()


# In[ ]:




