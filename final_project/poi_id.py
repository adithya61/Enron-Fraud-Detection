#!/usr/bin/python
# coding=utf-8
import pickle
import sys

import numpy as np
from itertools import compress
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Extract Features list
features_list = data_dict['METTS MARK'].keys()
features_pop_list = ['email_address', 'poi', 'other']
[features_list.remove(key) for key in features_pop_list]
features_list.insert(0, 'poi')

### Task 2: Remove outliers
del (data_dict['TOTAL'])

### Task 3: Create new feature(s)

# from message and to messages ratio
for key in data_dict:
    fro = data_dict[key]['from_this_person_to_poi']
    to = data_dict[key]['from_poi_to_this_person']
    if (fro not in ['NaN', '0', 0]) and (to not in ['NaN', '0', 0]):
        data_dict[key]['ratio'] = int(to) / int(fro)
    else:
        data_dict[key]['ratio'] = 'NaN'

features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')
features_list.remove('exercised_stock_options')
features_list.remove('restricted_stock_deferred')
features_list.remove('total_payments')
features_list.remove('deferral_payments')
features_list.remove('loan_advances')
features_list.remove('director_fees')
features_list.remove('deferred_income')
features_list.remove('to_messages')
features_list.remove('from_messages')

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

features = np.array(features)
labels = np.array(labels)


### Select Kbest
select = SelectKBest(f_classif, k=5)
features = select.fit_transform(features, labels)
features_mask = select.get_support()
print('Features used are : ', list(compress(features_list, features_mask)))
print('-----------------------------------------------------------------------------')

# X_train, X_test, y_train, y_test = train_test_split(features, labels)


### Task 4: Try a varity of classifiers

def clf_fit(features, labels ):


    kf = StratifiedKFold(n_splits=2)
    kf.get_n_splits(features, labels)
    clf_best = 0
    clf_list = []
    dt_clf = None
    rf_clf = None
    ada_clf = None
    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]


        # X_train, X_test, y_train, y_test = train_test_split(features, labels)

        print('DECISION TREE CLASSIFIER')
        print('----------------------------------------------------------------------')
        param_dt = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                      'max_features': ['auto', 'sqrt', 'log2', None],
                      'criterion': ['gini', 'entropy']}
        dt = DecisionTreeClassifier()
        dt_clf = GridSearchCV(dt, param_dt, scoring='f1')
        dt_clf.fit(X_train, y_train)
        dt_clf = dt_clf.best_estimator_
        dt_pred = dt_clf.predict(X_test)

        print('----------------------------------------------------------------------')
        print ('accuracy',accuracy_score(y_test, dt_pred))
        print ('recall', recall_score(y_test, dt_pred))
        print ('precision:', precision_score(y_test, dt_pred))
        print ('f1 score:', f1_score(y_test, dt_pred))
        print('----------------------------------------------------------------------')


        print('RANDOM FOREST CLASSIFIER')
        print('----------------------------------------------------------------------')
        param_rf = {
            'n_estimators': [2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 4, 8, 10],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
        rf = RandomForestClassifier(max_depth=2)
        rf_clf = GridSearchCV(rf, param_rf, scoring='f1')
        rf_clf.fit(X_train, y_train)
        rf_clf = rf_clf.best_estimator_
        rf_pred = rf_clf.predict(X_test)

        print ('accuracy',accuracy_score(y_test, rf_pred))
        print ('recall', recall_score(y_test, rf_pred))
        print ('precision:', precision_score(y_test, rf_pred))
        print ('f1 score:', f1_score(y_test, rf_pred))
        print('----------------------------------------------------------------------')

        print('ADA BOOST CLASSIFIER')
        print('----------------------------------------------------------------------')
        param_ada = {
            'n_estimators': [30, 50, 80, 100],
            'algorithm': ['SAMME', 'SAMME.R']
        }

        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        ada_clf = GridSearchCV(ada, param_ada, scoring='f1')
        ada_clf.fit(X_train, y_train)
        ada_clf = ada_clf.best_estimator_
        ada_pred = ada_clf.predict(X_test)

        print ('accuracy',accuracy_score(y_test, ada_pred))
        print ('recall', recall_score(y_test, ada_pred))
        print ('precision:', precision_score(y_test, ada_pred))
        print ('f1 score:', f1_score(y_test, ada_pred))
        print('----------------------------------------------------------------------')
    clf_list.append(dt_clf)
    clf_list.append(rf_clf)
    clf_list.append(ada_clf)
    return clf_list

clf = clf_fit(features, labels)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can

print('running on test set with 15k data points')

clf = clf[2]
dump_classifier_and_data(clf, my_dataset, features_list)

# for model in clf:
#     test_classifier(model, my_dataset, features_list)
