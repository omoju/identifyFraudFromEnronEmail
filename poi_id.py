
# coding: utf-8

# ## Identify Fraud from Enron Email
# #### Project Overview
# In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.
# 
# #### Project Introduction
# In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.
# 
# 

# In[1]:

#get_ipython().magic(u'pylab inline')


# In[2]:

import sys
from time import time

import numpy as np
import pickle

import matplotlib as pl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import fill_between
sns.set_style("white")


# In[3]:

dataPath = '/Users/omojumiller/mycode/MachineLearningNanoDegree/IntroToMachineLearning/'
sys.path.append(dataPath+'tools/')
sys.path.append(dataPath+'final_project/')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from helper_files import compareTwoFeatures, computeFraction, findPersonBasedOnTwoFeatures


# # Optimize Feature Selection/Engineering
# ## Task 1: Feature selection
# 
# The dataset used in this project is stored in a Python dictionary created by combining the Enron email and financial data, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.
# 
# financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
# 
# email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# 
# POI label: [‘poi’]
# 
# You can take a look at this [enron61702insiderpay.pdf](enron61702insiderpay.pdf) file to get a feel of the data yourself.
# 

# In[4]:

### Load the dictionary containing the dataset
### The data that I am loading in here is the one that has be cleansed of outliers. 
### For more information on that, refer to the notebook titled "cleanDataForOutliers" in the same folder.

with open(dataPath+'final_project/final_project_dataset.pkl', "r") as data_file:
    data_dict = pickle.load(data_file)
    

# Remove the source of the outlier

data_dict.pop( 'TOTAL')
data_dict.pop( 'BHATNAGAR SANJAY')

print "Deleted following records with keys:"
print 'TOTAL'
print 'BHATNAGAR SANJAY'



# ### Engineered Feature
# - #### Fraction of messages to and from POI

# ## Task 3: Create new feature(s)
# - features_list is a list of strings, each of which is a feature name.
# - The first feature must be "poi".
# - Store to `my_dataset` for easy export below.

# In[27]:

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    
    data_point["fraction_to_poi"] = fraction_to_poi
    


# Extract features and labels from dataset

# In[28]:

feature_list = ['poi', 'bonus', 'exercised_stock_options', 'restricted_stock']
#'fraction_from_poi','fraction_to_poi',
#'from_poi_to_this_person','from_this_person_to_poi','salary']

my_dataset = data_dict

data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "The features we are using to train our model are as follows:"
print feature_list


# ## Adhoc Features
# 
# ['poi', 'bonus', 'exercised_stock_options', 'restricted_stock']
#  #'fraction_from_poi','fraction_to_poi',
#  #'from_poi_to_this_person','from_this_person_to_poi','salary']
#  
#  ## XGBoost Features
# ['poi', 'expenses', 'other', 'deferred_income', 'from_poi_to_this_person', 'exercised_stock_options', 'salary', 'total_stock_value', 'total_payments', 'bonus', 'long_term_incentive', 'restricted_stock_deferred', 'from_messages', 'shared_receipt_with_poi', 'deferral_payments', 'to_messages']
# 
# ## Random Forest Features
# 
# ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'expenses', 'other', 'total_payments', 'restricted_stock', 'deferred_income', 'shared_receipt_with_poi', 'salary']

# In[29]:

import pandas as pd

data = pd.DataFrame({'poi':labels})
p = data.plot(kind = 'hist', rot = 0)
_ = p.set_xlabel('datapoint value'), p.set_ylabel("Frequency"), p.legend(["poi"])
_ = p.yaxis.tick_right()
_ = p.set_title('Histogram of POI values')


# # Pick and Tune an Algorithm
# ## Task 4: Try a variety of classifiers
# - Please name your classifier clf for easy export below.
# - Note that if you want to do PCA or other multi-stage operations, you'll need to use Pipelines. For more info: http://scikit-learn.org/stable/modules/pipeline.html
# 
# # Validate and Evaluate
# ## Task 5: Tune your classifier
# - Achieve better than .3 precision and recall. Using our testing script. Check the `tester.py` script in the final project folder for details on the evaluation method, especially the test_classifier function. Because of the small size of the dataset, the script uses `stratified shuffle split cross validation`. For more info: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 

# In[30]:

feature_list


# In[31]:

from sklearn.cross_validation import StratifiedShuffleSplit

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        
    return dict(clfname=clf.__class__.__name__, totalPrediction=total_predictions, accuracy=accuracy, 
                precision=precision,recall=recall, f1=f1, f2=f2,
                true_negatives=true_negatives, false_negatives=false_negatives,
                true_positives=true_positives,false_positives=false_positives)



# In[32]:

def print_Output_Table(results, prntLatex):
    """Print output table for classifier scores.

    Keyword arguments:
    clf -- classifier (scikit learn classifier model)
    precision -- int
    recall -- int
    accuracy -- int
    F1 -- array (float)
    prntLatex -- 1 for print optimized for latex, 0 for regular printing
    
    """
    
    print "{:^67}".format('Metrics')
    print "{:25}{:12}{:10}{:10}".format("", 'precision', 'recall', 'F1')
    print "-"*63

    if prntLatex:
        for i in range(0, len(results)):
            print "{:25}{:3}{:<12.3f}{:3}{:<10.3f}{:3}{:<10.3f}{:3}".format(results[i]['clfname'],'&', results[i]['precision'], 
                                                     '&', results[i]['recall'], '&', results[i]['f1'], '\\\\')
            
    print "\n\n"
    print "{:25}{:12}{:10}{:10}{:10}".format("", "TP", 'FP', 'TN', 'FN')
    print "-"*63

    if prntLatex:
        for i in range(0, len(results)):
            print "{:25}{:<12}{:<10}{:<10}{:<10}".format(results[i]['clfname'], results[i]['true_positives'], 
                                                     results[i]['false_positives'], results[i]['true_negatives'],
                                                            results[i]['false_negatives'])


# In[33]:

from sklearn.cross_validation import cross_val_score 
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier

folds = 1000


clf = ExtraTreesClassifier()
print '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"


t0 = time()
results_et = test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0)) 

cm_et = [[results_et['true_negatives'], results_et['false_negatives']],
     [results_et['true_positives'], results_et['false_positives']]]




# In[34]:


from sklearn.ensemble import RandomForestClassifier
from time import time


clf = RandomForestClassifier()
print '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"


t0 = time()
results_rf = test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))

cm_rf = [[results_rf['true_negatives'], results_rf['false_negatives']],
     [results_rf['true_positives'], results_rf['false_positives']]]




# In[44]:

from xgboost import XGBClassifier as XGBC


clf = XGBC()

print '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"


t0 = time()
results_xgb = test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))  

cm_xgb = [[results_xgb['true_negatives'], results_xgb['false_negatives']],
     [results_xgb['true_positives'], results_xgb['false_positives']]]




# In[36]:

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l1')

print '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"

t0 = time()
scores = cross_val_score(clf, features, labels, cv=10)
results_lr= test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))

cm_lr = [[results_lr['true_negatives'], results_lr['false_negatives']],
     [results_lr['true_positives'], results_lr['false_positives']]]




# In[37]:

results = [results_et, results_lr, results_rf, results_xgb]


# In[38]:

print_Output_Table(results, 1)


# ## Tune choosen algorithm

# In[90]:

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

print '_'*20, 'Tuning XGBClassifier', '_'*20
print "Training the data"


# max_delta_step, Maximum delta step we allow each tree's weight estimation to be. 
# If the value is set to 0, it means there is no constraint. If it is set to a positive value, 
# it can help making the update step more conservative. Usually this parameter is not needed, 
# but it might help in logistic regression when class is extremely imbalanced. 
# Set it to value of 1-10 might help control the update

cv_params = {'max_depth': [3,5,7],
             'min_child_weight': [1,3,5],
             'max_delta_step':[0, 1, 2],
             'learning_rate': [0.01, 0.1, 0.02, 0.2]}

params = {}
params['colsample_bytree'] = 1
params['objective'] = 'binary:logistic'

# Build a stratified shuffle object because of unbalanced data
folds = 1000
ssscv = StratifiedShuffleSplit(labels, folds, random_state = 42)


grid = GridSearchCV(XGBC(**params), 
                            cv_params, cv=ssscv, n_jobs = -1) 
grid.fit(features, labels)





# In[91]:

clf =  grid.best_estimator_
print clf
results_tuning = test_classifier(clf, my_dataset, feature_list, folds)

print "\n"
print '_'*20, 'Metrics of tuned '+clf.__class__.__name__, '_'*20, '\n'
print "precision : %.3f" % results_tuning['precision']
print "recall, : %.3f" % results_tuning['recall']
print "f1 : %.3f" % results_tuning['f1']


# XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#        gamma=0, learning_rate=0.02, max_delta_step=1, max_depth=3,
#        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=1)
# 
# 
# ____________________ Metrics of tuned XGBClassifier ____________________ 
# 
# precision : 0.639
# recall, : 0.318
# f1 : 0.424
# 

# ## Task 6: Export solution
# Dump your classifier, dataset, and features_list so anyone can check your results. You do not need to change anything below, but make sure that the version of `poi_id.py` that you submit can be run on its own and generates the necessary .pkl files for validating your results.

# In[93]:

dump_classifier_and_data(clf, my_dataset, feature_list)




