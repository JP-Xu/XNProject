from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import numpy as np
import pandas as pd
import regex as re
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_files(path, ext, string):
    """
    This function returns a list of files with specified extension and strings.
    """
    import os
    results=[]
    for file_ in os.scandir(path):
        if file_.is_file() and file_.name.endswith(ext) and string in file_.name:
            results.append(file_.name)
            
    return results


cancer_types = ['kidney', 'liver', 'pancreatic', 'bladder', 'ovarian', 'colon', 'breast', 'lung', 'leukemia']
Xy_files = get_files(path = './', ext = 'csv', string = 'Xy')

train_dict = {}
for cancer_ in cancer_types:
    train_dict[cancer_] = pd.read_csv('Xy_{}.csv'.format(cancer_), index_col = 0)

feature_importance_shuffle_results = {}
feature_importance_regular_results = {}

for cancer_ in cancer_types:
    feature_importance_regular_results[cancer_] = []
    X_pos = train_dict[cancer_].loc[train_dict[cancer_]['label'] == 'positive']
    for i in range(10):
        X_neg = train_dict[cancer_].loc[(train_dict[cancer_]['label'] == 'negative').index[i::10]]
        # define dataset
        X = pd.concat((X_pos.drop(columns=['label']), X_neg.drop(columns=['label'])), axis=0)
        y = pd.concat((X_pos['label'], X_neg['label']), axis=0)
        y_random = y.copy()
        feature_importance_shuffle_results[cancer_] = []
        # define the model
        model = RandomForestClassifier()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        feature_importance_regular_results[cancer_] += list(importance)
        for i in range(100):
            # shuffle labels
            y_random = y_random.sample(frac=1, ignore_index=True)
            # define the model
            model = RandomForestClassifier()
            # fit the model
            model.fit(X, y_random)
            # get importance
            importance = model.feature_importances_
            feature_importance_shuffle_results[cancer_] += list(importance)
            # summarize feature importance
            #for i,v in enumerate(importance):
            #     print('Feature: %0d, Score: %.5f' % (i,v))
            ## plot feature importance
            #pyplot.bar([x for x in range(len(importance))], importance)
            #pyplot.show()

zs = {}

for cancer_ in cancer_types:
    zs[cancer_] = []
    mean = np.average( np.array(feature_importance_shuffle_results[cancer_]).reshape(-1, 35), axis=0)
    std = np.std( np.array(feature_importance_shuffle_results[cancer_]).reshape(-1, 35), axis=0)
    z = abs(np.average( np.array(feature_importance_regular_results[cancer_]).reshape(-1, 35), axis=0) - mean) / std
    zs[cancer_].append(z)

print(zs)

import csv
csv_columns = ['Cancer','Features']
csv_file = "Train_dict.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        for k, v in zs.items():
            writer.writerow([k, v])
except IOError:
    print("I/O error")

