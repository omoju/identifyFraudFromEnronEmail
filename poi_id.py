
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
    
    


# In[5]:

#Stylistic Options for plots
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)


# ## Outlier removal
# If there are outliers, remove outliers
# 
# This is an iteratable process. I need to do this for each combination of features I want to use
# 

# In[ ]:




# In[6]:

plt.rcParams["axes.labelsize"] = 16.
plt.rcParams["xtick.labelsize"] = 14.
plt.rcParams["ytick.labelsize"] = 14.
plt.rcParams["legend.fontsize"] = 11.
plt.rcParams["axes.titlesize"] = 1.25 * plt.rcParams['font.size']
plt.rcParams['figure.figsize'] = (8, 6)


# In[7]:

data = compareTwoFeatures('salary', 'bonus', data_dict, "SALARY versus BONUS")
print "Clean data for outliers"


# In[8]:

salary = featureFormat(data_dict, ['salary'], remove_any_zeroes=True)




ax = plt.plot(salary, label='Salary', color=tableau20[0])


_= plt.xlabel('datapoint value')
_= plt.title('Salary')

#_= plt.legend(loc='upper center', shadow=True, fontsize='medium')
_= plt.legend()
plt.show()


# In[ ]:




# In[9]:

np.where(data > 0.8 * 1e8) # This is where the outlier is, what I have to do now is find out who it is


# In[10]:

data[57] # So whose bonus is 97343619?
# What’s the name of the dictionary key of this data point?


# In[11]:

for key, value in data_dict.iteritems():
    if (value['bonus'] >= int(data[57][1]) and 
        value['bonus'] != "NaN" and
        value['salary'] != "NaN"):
        print "{:20}{:12}${:<12,.2f}{:12}${:<12,.2f}".format(key, 'salary is ', value['salary'],
                                                   ' bonus ', value['bonus'])
        
    if (value['restricted_stock'] < 0):
        print "{:20}{:12}{:12}".format(key, 'restricted_stock is ', value['restricted_stock'])




# In[12]:

# Remove the source of the outlier

data_dict.pop( 'TOTAL')
data_dict.pop( 'BHATNAGAR SANJAY')

print "Deleted following records with keys:"
print 'TOTAL'
print 'BHATNAGAR SANJAY'

# We can now go back and rerun the regression to see what the data really looks like.


# In[13]:

data = compareTwoFeatures('salary', 'bonus', data_dict, "SALARY versus BONUS cleansed of outliers")


# ## Task: Data exploration
# - Get descriptive statistics
# 
# 

# In[14]:

## Creating a pandas dataframe so that we can easily get descriptive statistics about our features

import itertools


salary = featureFormat(data_dict, ['salary'], remove_any_zeroes=True)
bonus = featureFormat(data_dict, ['bonus'], remove_any_zeroes=True)
exerStockOptions = featureFormat(data_dict, ['exercised_stock_options'], remove_any_zeroes=True)
restrictedStock = featureFormat(data_dict, ['restricted_stock'], remove_any_zeroes=True)

bonus = list(itertools.chain.from_iterable(bonus))
salary = list(itertools.chain.from_iterable(salary))
exerStockOptions = list(itertools.chain.from_iterable(exerStockOptions))
restrictedStock = list(itertools.chain.from_iterable(restrictedStock))




# In[15]:

## Pad feature list with zeros to ensure all columns have equal lenght
## Otherwise we won't be able to transfor the individual feature list into a dataframe

size = len(restrictedStock) - len(bonus)
temp = [0.0] * size 


bonus = bonus + temp

size = len(restrictedStock) - len(salary)
temp = [0.0] * size 

salary = salary + temp


size = len(restrictedStock) - len(exerStockOptions)
temp = [0.0] * size 

exerStockOptions = exerStockOptions + temp


# In[16]:

import pandas as pd


df = pd.DataFrame({'salary': salary, 'bonus': bonus, 'exercisedStockOptions': exerStockOptions, 
                   'restrictedStock': restrictedStock})
x = range(0,len(df))

pltSalary = df['salary']

ax = plt.plot(pltSalary, label='Salary', color=tableau20[0])
fill_between(x, pltSalary, 0, alpha=0.5, color=tableau20[19])

_= plt.xlabel('datapoint value')
_= plt.title('Salary')

#_= plt.legend(loc='upper center', shadow=True, fontsize='medium')
_= plt.legend()
plt.xlim(0, 94)
plt.show()

df['salary'].describe()


# In[ ]:




# In[17]:



plt.plot(x, df['exercisedStockOptions'], label='Exercised Stock Options', color=tableau20[8])
plt.plot(x, df['restrictedStock'], label='Restricted Stock', color=tableau20[0])
#plt.plot(x, df['bonus'], label='Bonus', color=tableau20[4])


fill_between(x, df['exercisedStockOptions'], 0, alpha=0.5, color=tableau20[19])
fill_between(x, df['restrictedStock'], 0, alpha=0.5, color=tableau20[4])
#fill_between(x, df['bonus'], 0, alpha=0.5,color=tableau20[4])


_= plt.xlabel('datapoint value')
_= plt.title('')

#_= plt.legend(loc='upper center', shadow=True, fontsize='medium')
_= plt.legend(bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=1, fontsize='large')
plt.show()


# In[18]:

x = range(0,len(df))

plt.plot(x, df['exercisedStockOptions'], label='Exercised Stock Options', color=tableau20[8])
fill_between(x, df['exercisedStockOptions'], 0, alpha=0.5, color=tableau20[19])

_= plt.xlabel('datapoint value')
_= plt.title('Exercised Stocks')

_= plt.legend()
plt.xlim(0, 100)
plt.show()

df['exercisedStockOptions'].describe()


# In[19]:

x = range(0,len(df))

plt.plot(x, df['restrictedStock'], label='Restricted Stock', color=tableau20[0])
fill_between(x, df['restrictedStock'], 0, alpha=0.5, color=tableau20[4])

_= plt.xlabel('datapoint value')
_= plt.title('Restricted Stock')

_= plt.legend()

plt.xlim(0, 108)
plt.show()

df['restrictedStock'].describe()


# In[20]:

x = range(0,len(df))

plt.plot(x, df['bonus'], label='Bonus', color=tableau20[0])
fill_between(x, df['bonus'], 0, alpha=0.5, color=tableau20[4])

_= plt.xlabel('datapoint value')
_= plt.title('Bonus')
plt.xlim(0, 81)
_= plt.legend()

df['bonus'].describe()


# In[21]:

def printLatex(feature1, feature2, the_data_dict, treshold):
    def getKey(item):
        return item[2]

    temp = []
    for key, value in the_data_dict.iteritems():
        if (value[feature1] != "NaN") and (value[feature2] != "NaN" and value[feature2] > treshold):
            temp.append(( key, value[feature1], value[feature2]))
            
    ### print out in ascending order of feature2    
    temp = sorted(temp, key=getKey)
    print "{:20}{:3}{:14}{:3}{:12}{:3}".format("Name".upper(), '&', feature1.upper(), '&', feature2.upper(), '\\\\')
    
    for item in temp:    
        print "{:20}{:3}\${:<14,.2f}{:3}{:12,}{:3}".format(item[0], '&', item[1], '&', item[2], '\\\\')
           
    


# In[22]:

f1, f2 = 'salary','exercised_stock_options'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus EXERCISED STOCK OPTIONS")


# In[23]:

treshold = 8000000
#printLatex(f1, f2, data_dict, treshold)


# In[24]:

f1, f2 = 'salary','restricted_stock'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus RESTRICTED STOCK")



# In[25]:

treshold = 3000000
#printLatex(f1, f2, data_dict, treshold)


# In[26]:

f1, f2 = 'salary','bonus'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus BONUS")


# In[27]:

treshold = 4000000
#printLatex(f1, f2, data_dict, treshold)


# In[28]:

data_dict['LAVORATO JOHN J']


# ### Engineered Feature
# - #### Fraction of messages to and from POI

# ## Task 3: Create new feature(s)
# - features_list is a list of strings, each of which is a feature name.
# - The first feature must be "poi".
# - Store to `my_dataset` for easy export below.

# In[29]:

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print'{:5}{:35}{:.2f}'.format('FROM ', name, fraction_from_poi)
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    #print'{:5}{:35}{:.2f}'.format('TO: ', name, fraction_to_poi)
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    
    data_point["fraction_to_poi"] = fraction_to_poi
    


# In[30]:

f1, f2 = 'bonus','exercised_stock_options'
data = compareTwoFeatures(f1, f2, data_dict, 'Bonus v Exercised Stock Options')


# Extract features and labels from dataset

# In[142]:

feature_list = ['poi', 'bonus', 'exercised_stock_options', 'restricted_stock']


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

# In[143]:

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

# In[144]:

feature_list


# In[145]:

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



# In[146]:

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

print cm_et


# In[147]:


from sklearn.ensemble import RandomForestClassifier
from time import time


clf = RandomForestClassifier()
print '\n', '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"


t0 = time()
results_rf = test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))

cm_rf = [[results_rf['true_negatives'], results_rf['false_negatives']],
     [results_rf['true_positives'], results_rf['false_positives']]]

print cm_rf


# In[148]:

from xgboost import XGBClassifier as XGBC

clf = XGBC()
print '\n', '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"


t0 = time()
results_xgb = test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))  

cm_xgb = [[results_xgb['true_negatives'], results_xgb['false_negatives']],
     [results_xgb['true_positives'], results_xgb['false_positives']]]

print cm_xgb


# In[149]:

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l1')

print '\n', '_'*20, clf.__class__.__name__, '_'*20
print "Training the data"

t0 = time()
scores = cross_val_score(clf, features, labels, cv=10)
results_lr= test_classifier(clf, my_dataset, feature_list, folds)
print("done in %0.3fs" % (time() - t0))

cm_lr = [[results_lr['true_negatives'], results_lr['false_negatives']],
     [results_lr['true_positives'], results_lr['false_positives']]]

print cm_lr


# In[150]:

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


# In[151]:

results = [results_et, results_lr, results_rf, results_xgb]


# In[152]:

print_Output_Table(results, 1)


# ## Tune choosen algorithm

# In[153]:



from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

print '\n', '_'*20, 'Tuning LogisticRegression', '_'*20
print "Training the data"

clf = LogisticRegression( penalty='l1', solver='liblinear') 

# Build a stratified shuffle object because of unbalanced data
folds = 1000
ssscv = StratifiedShuffleSplit(labels, folds, random_state = 42)

# For an initial search, a logarithmic grid with basis 10 is often helpful. 
# Using a basis of 2, a finer tuning can be achieved but at a much higher cost.

C_range = 2. ** np.arange(-3, 2)
tolerance = [1e-2, 1e-3, 1e-4]

param_grid = dict(tol=tolerance, C=C_range)  

grid = GridSearchCV(clf, param_grid=param_grid, cv=ssscv)
grid.fit(features, labels)



# In[164]:

clf =  grid.best_estimator_
print clf

results_tuning = test_classifier(clf, my_dataset, feature_list, folds)

print "\n"
print '\n', '_'*20, 'Metrics of tuned '+clf.__class__.__name__, '_'*20, '\n'
for item in results_tuning:
    print item, ": ", results_tuning[item]


# LogisticRegression(C=0.125, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)
#           
#           
# 	Accuracy: 0.86438	Precision: 0.73010	Recall: 0.18800	F1: 0.29901	F2: 0.22079
# 	Total predictions: 13000	True positives:  376	False positives:  139	False negatives: 1624	True negatives: 10861

# ## Task 6: Export solution
# Dump your classifier, dataset, and features_list so anyone can check your results. You do not need to change anything below, but make sure that the version of `poi_id.py` that you submit can be run on its own and generates the necessary .pkl files for validating your results.

# In[165]:

dump_classifier_and_data(clf, my_dataset, feature_list)


# In[ ]:




# In[ ]:



