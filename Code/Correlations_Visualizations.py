#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

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


class correlations(object):
    
    def __init__(self, df):
        self.df = df
        
    def infer_data_types(self):
        data = self.df
        numerical_cols = []
        categorical_cols = []
        for col in data.columns:
            if data[col].dtype != object:  # Exclude strings
                if len(data[col].unique())<10:
                    categorical_cols.append(col)
                    data[col] = data[col].astype('str')
                else:
                    numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        # Print final result
        print("___MEMORY USAGE:___")
        mem_usg = data.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        return data, categorical_cols,numerical_cols
    
    def data_stats(self):
        data = self.df
        data, categorical_cols,numerical_cols = self.infer_data_types()
        print("Number of Rows:",len(data))
        print("Number of Columns:",data.shape[1])
        print("Number of Numerical Columns:",len(numerical_cols))
        print("Number of Categorical Columns:",len(categorical_cols))
        col_stats = {}
        for col in numerical_cols:
            stats = {}
            stats['col_type'] = 'Numerical'
            stats['num_unique'] = len(data[col].unique())
            stats['num_missing']= sum(data[col].isin([np.nan]))
            stats['min']= min(data[col])
            stats['max']= max(data[col])
            stats['mean']= np.mean(data[col])
            stats['mean']= np.mean(data[col])
            stats['std']= np.std(data[col])
            col_stats[col] = stats

        for col in categorical_cols:
            stats = {}
            stats['col_type'] = 'Categorical'
            stats['num_unique'] = len(data[col].unique())
            stats['num_missing']= sum(data[col].isin([np.nan]))
            stats['col_mode']= data[col].mode()[0]
            col_stats[col] = stats   
        return data, col_stats,categorical_cols,numerical_cols
    
    def convert(self, to):
        data = self.df
        converted = None
        if to == 'array':
            if isinstance(data, np.ndarray):
                converted = data
            elif isinstance(data, pd.Series):
                converted = data.values
            elif isinstance(data, list):
                converted = np.array(data)
            elif isinstance(data, pd.DataFrame):
                converted = data.as_matrix()
        elif to == 'list':
            if isinstance(data, list):
                converted = data
            elif isinstance(data, pd.Series):
                converted = data.values.tolist()
            elif isinstance(data, np.ndarray):
                converted = data.tolist()
        elif to == 'dataframe':
            if isinstance(data, pd.DataFrame):
                converted = data
            elif isinstance(data, np.ndarray):
                converted = pd.DataFrame(data)
        else:
            raise ValueError("Unknown data conversion: {}".format(to))
        if converted is None:
            raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
        else:
            return converted
    
    def conditional_entropy(self, x, y):
        """
        Calculates the conditional entropy of x given y: S(x|y)
        Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
        :param x: list / NumPy ndarray / Pandas Series
            A sequence of measurements
        :param y: list / NumPy ndarray / Pandas Series
            A sequence of measurements
        :return: float
        """
        # entropy of x given y
        data = self.df
        y_counter = Counter(data[y])
        xy_counter = Counter(list(zip(data[x], data[y])))
        total_occurrences = sum(y_counter.values())
        entropy = 0.0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y/p_xy)
        return entropy
    
    def cramers_v(self, x, y):
        data = self.df
        confusion_matrix = pd.crosstab(data[x], data[y])
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

    def theils_u(self, x, y):
        data = self.df
        s_xy = self.conditional_entropy(x, y)
        x_counter = Counter(data[x])
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
        s_x = ss.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x

    def correlation_ratio(self, x, y):
        data = self.df
        categories = data[x]
        measurements = data[y]
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta
    
    def associations(self, nominal_columns = None, mark_columns = False, theil_u = False, plot = True,
                          return_results = False, **kwargs):
        """
        Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
        continuous features using:
         - Pearson's R for continuous-continuous cases
         - Correlation Ratio for categorical-continuous cases
         - Cramer's V or Theil's U for categorical-categorical cases
        :param dataset: NumPy ndarray / Pandas DataFrame
            The data-set for which the features' correlation is computed
        :param nominal_columns: string / list / NumPy ndarray
            Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
            columns are categorical, or None (default) to state none are categorical
        :param mark_columns: Boolean (default: False)
            if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
            continuous), as provided by nominal_columns
        :param theil_u: Boolean (default: False)
            In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
        :param plot: Boolean (default: True)
            If True, plot a heat-map of the correlation matrix
        :param return_results: Boolean (default: False)
            If True, the function will return a Pandas DataFrame of the computed associations
        :param kwargs:
            Arguments to be passed to used function and methods
        :return: Pandas DataFrame
            A DataFrame of the correlation/strength-of-association between all features
        """
        
        dataset = self.df
        dataset = self.convert('dataframe')
        columns = dataset.columns
        if nominal_columns is None:
            nominal_columns = list()
        elif nominal_columns == 'all':
            nominal_columns = columns
        corr = pd.DataFrame(index = columns, columns = columns)
        for i in range(0,len(columns)):
            for j in range(i,len(columns)):
                if i == j:
                    corr[columns[i]][columns[j]] = 1.0
                else:
                    if columns[i] in nominal_columns:
                        if columns[j] in nominal_columns:
                            if theil_u:
                                corr[columns[j]][columns[i]] = self.theils_u(columns[i], columns[j])
                                corr[columns[i]][columns[j]] = self.theils_u(columns[j], columns[i])
                            else:
                                cell = self.cramers_v(columns[i], columns[j])
                                corr[columns[i]][columns[j]] = cell
                                corr[columns[j]][columns[i]] = cell
                        else:
                            cell = self.correlation_ratio(columns[i], columns[j])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        if columns[j] in nominal_columns:
                            cell = self.correlation_ratio(columns[j], columns[i])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                        else:
                            cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
        corr.fillna(value=np.nan, inplace=True)
        if mark_columns:
            marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
            corr.columns = marked_columns
            corr.index = marked_columns
        if plot:
            plt.figure(figsize = (20,20))#kwargs.get('figsize',None))
            sns.heatmap(corr, annot = kwargs.get('annot',True), fmt = kwargs.get('fmt','.2f'), cmap = 'coolwarm')
            plt.show()
        if return_results:
            return corr
        
    def find_top_correlated_cols(self, results, target_column, n = 5, mark_columns = False):
        if mark_columns:
            target_corr_df = results.reset_index()[['index', target_column+' (nom)']]
        else:
            target_corr_df = results.reset_index()[['index', target_column]]
        target_corr_df.columns = ['variable','corr']
        if mark_columns:
            target_corr_df = target_corr_df[target_corr_df['variable'] != target_column+' (nom)']
        else:
            target_corr_df = target_corr_df[target_corr_df['variable'] != target_column]
        target_corr_df = target_corr_df.sort_values(by = 'corr', ascending = False)
        var_corr = target_corr_df.head(n).set_index('variable').to_dict()['corr']
        print("top "+str(n)+" correlated columns are:")
        results = []
        for k,v in var_corr.items():
            results.append([k, target_column, v])
            print(k,v)
            
        results = pd.DataFrame(results, columns = ['Var_x', 'Var_y', 'Correlation'])
        results = results.sort_values('Correlation', ascending = False)
        return results
    
    def highly_correlated_columns(self, results, target_column, cut_off = 0.4, mark_columns = False):
        if mark_columns:
            res =  results[[x for x in results.columns if x!=target_column+" (nom)"]]
        else:
            res =  results[[x for x in results.columns if x!=target_column]]
        col_corr = set()
        results = []
        for i in range(len(res.columns)):
            for j in range(i):
                if res.iloc[i, j] >= cut_off:
                    print("Column "+res.columns[i]+" correlated with "+res.columns[j])
                    colname = res.columns[i]
                    col_corr.add(colname)
                    results.append([res.columns[i], res.columns[j], res.iloc[i, j]])
        results = pd.DataFrame(results, columns = ['Var_x', 'Var_y', 'Correlation'])
        results = results.sort_values('Correlation', ascending = False)
        return results
        


class visualization(object):
    
    def __init__(self, df):
        self.df = df
        self.correlations = correlations(self.df)
        _, _, self.categorical_cols, self.numerical_cols = self.correlations.data_stats()
        self.correlation_matrix = self.correlations.associations(nominal_columns = self.categorical_cols,                                                             return_results = True, plot = False)
        
    def create_boxplot(self, x, y):
        data = self.df
        sns.set(style = "ticks", palette = "pastel")
        sns.boxplot(x = x, y = y, data = data)
        sns.despine(offset = 10, trim = True)
        
    def create_stacked(self, x, y):
        data = self.df
        sns.set(style = "ticks", palette = "pastel")
        p_table = pd.pivot_table(data, index = x, columns = y, aggfunc = 'size')
        p_table = p_table.div(p_table.sum(axis = 1), axis = 0)
        p_table.plot.bar(stacked = True)
        sns.despine()
        
    def create_scatter_plot_matrix(self, target_column = None):
        data = self.df
        if target_column is None:
            sns.pairplot(data)
        else:
            sns.pairplot(data, hue = target_column, diag_kind = 'auto')
            
    def create_scatter_plot(self, x, y):
        data = self.df
        sns.scatterplot(data[x], data[y])
        
    def see_outlier(self, col):
        data = self.df
        z = np.abs(stats.zscore(data[col]))
        print("Some Outlier Values are:")
        locs = np.where(z > 3)[0]

        for i,loc in enumerate(locs):
            if i < 5:
                print(data[col].values[loc])
        sns.boxplot(x = data[col])
        
    def plot_highly_correlated(self, target_column, with_target = False):
        if with_target:
            high_correlated_vars = self.correlations.find_top_correlated_cols(self.correlation_matrix, target_column)
        else:
            high_correlated_vars = self.correlations.highly_correlated_columns(self.correlation_matrix, target_column)

        for i in range(high_correlated_vars.shape[0]):
            if (high_correlated_vars['Var_x'][i] in self.categorical_cols) & (high_correlated_vars['Var_y'][i] in self.categorical_cols):
                self.create_stacked(high_correlated_vars['Var_x'][i], high_correlated_vars['Var_y'][i])
                plt.show()
            elif (high_correlated_vars['Var_x'][i] in self.categorical_cols) & (high_correlated_vars['Var_y'][i] in self.numerical_cols):
                self.create_boxplot(high_correlated_vars['Var_x'][i], high_correlated_vars['Var_y'][i])
                plt.show()
            elif (high_correlated_vars['Var_x'][i] in self.numerical_cols) & (high_correlated_vars['Var_y'][i] in self.categorical_cols):
                self.create_boxplot(high_correlated_vars['Var_y'][i], high_correlated_vars['Var_x'][i])
                plt.show()
            elif (high_correlated_vars['Var_x'][i] in self.numerical_cols) & (high_correlated_vars['Var_y'][i] in self.numerical_cols):
                self.create_scatter_plot(high_correlated_vars['Var_x'][i], high_correlated_vars['Var_y'][i])
                plt.show()