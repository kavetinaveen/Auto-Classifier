import scipy.stats as ss
from collections import Counter
import math 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import yaml
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Correlations_Visualizations import correlations
from Feature_Engineering import transformations_interactions
from Feature_Selection import feature_selection

class Classification(object):
    
    def __init__(self, params):
        self.params = params
        self.df = pd.read_csv(self.params['info']['base_dir'] + self.params['info']['file_location'])
        self.cor = correlations(self.df)
        _, _, self.categorical_cols, self.numerical_cols =  self.cor.data_stats()
        
#         if self.params['feature_engineering']['transformations']['is_transformations']:
        transformations = self.params['feature_engineering']['transformations']['transformations_list']
        
#         if self.params['feature_engineering']['interactions']['is_interactions']:
        interactions = self.params['feature_engineering']['interactions']['interactions_list']
        info_features = [self.params['cross_validation']['date_column'], self.params['info']['id_column']]
        
        self.feat_selection = feature_selection(self.df, self.params['model']['dep_var'], \
                                                self.params['feature_selection']['top_k_best'], \
                                                transformations, \
                                                interactions,
                                               info_features)
        
        self.df_processed = self.feat_selection.process()
    
    def split_data_into_train_test(self):
        df = self.df_processed
        y = df[self.params['model']['dep_var']]
        X_train, X_test, y_train, y_test = train_test_split(df, y,\
                                                            train_size = 1 - self.params['cross_validation']['test_split'],\
                                                            test_size = self.params['cross_validation']['test_split'])
        del X_train[self.params['model']['dep_var']]
        del X_test[self.params['model']['dep_var']]
        return X_train, X_test, y_train, y_test
    
    def split_data_into_train_test_time_based(self):
        if self.params['cross_validation']['date_column'] is None:
            raise Exception('Time based train/test split selected but date column not provided')
        df = self.df_processed
        date_column = self.params['cross_validation']['date_column']
        id_column = self.params['info']['id_column']
        df[date_column] = pd.to_datetime(df[date_column])
        train_start = pd.to_datetime(self.params['cross_validation']['training_start'])
        train_end = pd.to_datetime(self.params['cross_validation']['training_end'])
        test_start = pd.to_datetime(self.params['cross_validation']['testing_start'])
        test_end = pd.to_datetime(self.params['cross_validation']['testing_end'])
        
        X_train = df[(df[date_column] >= train_start) & (df[date_column] <= train_end)]
        y_train = X_train[self.params['model']['dep_var']]
        
        X_test = df[(df[date_column] >= test_start) & (df[date_column] <= test_end)]
        y_test = X_test[self.params['model']['dep_var']]
        
        self.id_vals_train = X_train[id_column]
        self.id_vals_test = X_test[id_column]
        
        self.date_vals_train = X_train[date_column]
        self.date_vals_test = X_test[date_column]
        
        del X_train[self.params['model']['dep_var']]
        del X_test[self.params['model']['dep_var']]
        
        del X_train[date_column]
        del X_test[date_column]
        
        del X_train[id_column]
        del X_test[id_column]
        
        return X_train, X_test, y_train, y_test
    
    def logistic_regression(self):
        if not 'lr' in self.params['model']['model_list']:
            raise Exception('Logistic Regression not listed in the model list parameter')
        space = self.params
        clf = LogisticRegression(penalty = self.params['model']['lr_params']['penalty'],\
                                C = self.params['model']['lr_params']['C'],\
                                fit_intercept = self.params['model']['lr_params']['fit_intercept'],\
                                class_weight = self.params['model']['lr_params']['class_weight'],\
                                solver = self.params['model']['lr_params']['solver'],\
                                max_iter = self.params['model']['lr_params']['max_iter'],\
                                multi_class = self.params['model']['lr_params']['multi_class'])
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')
            
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
            
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_lr.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_lr_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_lr_train.csv', index = False)
        
        print('Saved trained model (Logistic Regression): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_lr.pkl'))
        print('Written test predictions (Logistic Regression): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_lr_test.csv'))
        print('Written train predictions (Logistic Regression): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_lr_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK}
    
    def random_forest(self):
        if not 'rf' in self.params['model']['model_list']:
            raise Exception('Random Forest not listed in the model list parameter')
        space = self.params
        
        clf = RandomForestClassifier(n_estimators=self.params['model']['rf_params']['n_estimators'],
                criterion=self.params['model']['rf_params']['criterion'],
                max_depth=self.params['model']['rf_params']['max_depth'],
                min_samples_split=self.params['model']['rf_params']['min_samples_split'],
                min_samples_leaf=self.params['model']['rf_params']['min_samples_leaf'],
                min_weight_fraction_leaf=self.params['model']['rf_params']['min_weight_fraction_leaf'],
                max_features=self.params['model']['rf_params']['max_features'],
                max_leaf_nodes=self.params['model']['rf_params']['max_leaf_nodes'],
                min_impurity_decrease=self.params['model']['rf_params']['min_impurity_decrease'],
                class_weight=self.params['model']['rf_params']['class_weight'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()
            
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/') 
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_rf.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_rf_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_rf_train.csv', index = False)
        
        print('Saved trained model (Random Forest): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_rf.pkl'))
        print('Written test predictions (Random Forest): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_rf_test.csv'))
        print('Written train predictions (Random Forest): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_rf_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
    
    def decision_tree(self):
        if not 'dt' in self.params['model']['model_list']:
            raise Exception('Decision Tree not listed in the model list parameter')
        space = self.params
        
        clf = DecisionTreeClassifier(max_depth=self.params['model']['dt_params']['max_depth'],
                min_samples_split=self.params['model']['dt_params']['min_samples_split'],
                min_samples_leaf=self.params['model']['dt_params']['min_samples_leaf'],
                min_weight_fraction_leaf=self.params['model']['dt_params']['min_weight_fraction_leaf'],
                max_features=self.params['model']['dt_params']['max_features'],
                max_leaf_nodes=self.params['model']['dt_params']['max_leaf_nodes'],
                class_weight=self.params['model']['dt_params']['class_weight'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_dt.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_dt_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_dt_train.csv', index = False)
        
        print('Saved trained model (Decision Tree): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_dt.pkl'))
        print('Written test predictions (Decision Tree): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_dt_test.csv'))
        print('Written train predictions (Decision Tree): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_dt_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
        
        
    def extra_tree(self):
        if not 'et' in self.params['model']['model_list']:
            raise Exception('Extra Tree Classifier not listed in the model list parameter')
        space = self.params
        
        clf = ExtraTreesClassifier(n_estimators=self.params['model']['rf_params']['n_estimators'],
                                   max_depth=self.params['model']['dt_params']['max_depth'],
                                    min_samples_split=self.params['model']['dt_params']['min_samples_split'],
                                    min_samples_leaf=self.params['model']['dt_params']['min_samples_leaf'],
                                    min_weight_fraction_leaf=self.params['model']['dt_params']['min_weight_fraction_leaf'],
                                    max_features=self.params['model']['dt_params']['max_features'],
                                    max_leaf_nodes=self.params['model']['dt_params']['max_leaf_nodes'],
                                   min_impurity_decrease=self.params['model']['rf_params']['min_impurity_decrease'],
                                    class_weight=self.params['model']['dt_params']['class_weight'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_et.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_et_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_et_train.csv', index = False)
        
        print('Saved trained model (Extra Trees): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_et.pkl'))
        print('Written test predictions (Extra Trees): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_et_test.csv'))
        print('Written train predictions (Extra Trees): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_et_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
        
    def neural_net(self):
        if not 'nn' in self.params['model']['model_list']:
            raise Exception('NN Classifier not listed in the model list parameter')
        space = self.params
        
        clf = MLPClassifier(hidden_layer_sizes=self.params['model']['nn_params']['hidden_layer_sizes'],
                                   learning_rate_init=self.params['model']['nn_params']['learning_rate_init'],
                                    activation=self.params['model']['nn_params']['activation'],
                                    solver=self.params['model']['nn_params']['solver'])

        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_nn.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_nn_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_nn_train.csv', index = False)
        
        print('Saved trained model (Neural Net): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_nn.pkl'))
        print('Written test predictions (Neural Net): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_nn_test.csv'))
        print('Written train predictions (Neural Net): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_nn_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
    
    def adaboost_classifier(self):
        if not 'ac' in self.params['model']['model_list']:
            raise Exception('ADA Classifier not listed in the model list parameter')
        space = self.params
        
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                         n_estimators=self.params['model']['ac_params']['n_estimators'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_ac.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_ac_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_ac_train.csv', index = False)
        
        print('Saved trained model (Adaboost Classifier): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_ac.pkl'))
        print('Written test predictions (Adaboost Classifier): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_ac_test.csv'))
        print('Written train predictions (Adaboost Classifier): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_ac_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
    
    def gradient_boosting_classifier(self):
        if not 'gb' in self.params['model']['model_list']:
            raise Exception('GB Classifier not listed in the model list parameter')
        space = self.params
        
        clf = GradientBoostingClassifier(
                learning_rate=self.params['model']['gb_params']['learning_rate'],
                n_estimators=self.params['model']['gb_params']['n_estimators'],
                criterion=self.params['model']['gb_params']['criterion'],
                subsample=self.params['model']['gb_params']['subsample'],
                max_depth=self.params['model']['gb_params']['max_depth'],
                min_samples_split=self.params['model']['gb_params']['min_samples_split'],
                min_samples_leaf=self.params['model']['gb_params']['min_samples_leaf'],
                min_weight_fraction_leaf=self.params['model']['gb_params']['min_weight_fraction_leaf'],
                max_features=self.params['model']['gb_params']['max_features'],
                max_leaf_nodes=self.params['model']['gb_params']['max_leaf_nodes'],
                min_impurity_decrease=self.params['model']['gb_params']['min_impurity_decrease'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        

        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_gb.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_gb_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_gb_train.csv', index = False)
        
        print('Saved trained model (Gradient Boosting): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_gb.pkl'))
        print('Written test predictions (Gradient Boosting): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_gb_test.csv'))
        print('Written train predictions (Gradient Boosting): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_gb_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
        
    def knn_classifier(self):
        if not 'kn' in self.params['model']['model_list']:
            raise Exception('KNN Classifier not listed in the model list parameter')
        space = self.params
        
        clf = KNeighborsClassifier(
                n_neighbors=self.params['model']['kn_params']['n_neighbors'],
                metric=self.params['model']['kn_params']['metric'],
                leaf_size=self.params['model']['kn_params']['leaf_size'],
                p=self.params['model']['kn_params']['p'])
        
        if self.params['cross_validation']['time_based_test_split']:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test_time_based()
        else:
            X_train, X_test, y_train, y_test = self.split_data_into_train_test()

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        pred_prob = [x[1] for x in pred_prob]
        predictions = pd.DataFrame({'id': self.id_vals_test, 'date': self.date_vals_test, 'prob': pred_prob, 'status': y_test})
        
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_train = [x[1] for x in pred_prob_train]
        predictions_train = pd.DataFrame({'id': self.id_vals_train, 'date': self.date_vals_train, 'prob': pred_prob_train, 'status': y_train})        
        
        if not os.path.exists(self.params['info']['base_dir']+'models/'):
            os.mkdir(self.params['info']['base_dir']+'models/')  
        
        if not os.path.exists(self.params['info']['base_dir']+'predictions/'):
            os.mkdir(self.params['info']['base_dir']+'predictions/')
        
        joblib.dump(clf, self.params['info']['base_dir'] + 'models/' + 'trained_model_kn.pkl')
        predictions.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_kn_test.csv', index = False)
        predictions_train.to_csv(self.params['info']['base_dir']+'predictions/'+'predictions_kn_train.csv', index = False)
        
        print('Saved trained model (KNN): {}'.format(self.params['info']['base_dir'] + 'models/' + 'trained_model_kn.pkl'))
        print('Written test predictions (KNN): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_kn_test.csv'))
        print('Written train predictions (KNN): {}'.format(self.params['info']['base_dir']+'predictions/'+'predictions_kn_train.csv'))
        
        accuracy = accuracy_score(y_test, pred)
        return {'loss':-accuracy, 'status': STATUS_OK }
    
