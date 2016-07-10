
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
import matplotlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pickle


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
    
    


# ## Outlier removal
# If there are outliers, remove outliers
# 
# This is an iteratable process. I need to do this for each combination of features I want to use
# 

# In[5]:

data = compareTwoFeatures('salary', 'bonus', data_dict, "SALARY versus BONUS")
print "Clean data for outliers"


# In[6]:

np.where(data > 0.8 * 1e8) # This is where the outlier is, what I have to do now is find out who it is


# In[7]:

data[57] # So whose bonus is 97343619?
# What’s the name of the dictionary key of this data point?


# In[8]:

for key, value in data_dict.iteritems():
    if (value['bonus'] >= int(data[57][1]) and 
        value['bonus'] != "NaN" and
        value['salary'] != "NaN"):
        print "{:20}{:12}${:<12,.2f}{:12}${:<12,.2f}".format(key, 'salary is ', value['salary'],
                                                   ' bonus ', value['bonus'])
        
    if (value['restricted_stock'] < 0):
        print "{:20}{:12}{:12}".format(key, 'restricted_stock is ', value['restricted_stock'])




# In[9]:

# Remove the source of the outlier

data_dict.pop( 'TOTAL')
data_dict.pop( 'BHATNAGAR SANJAY')

print "Deleted following records with keys:"
print 'TOTAL'
print 'BHATNAGAR SANJAY'

# We can now go back and rerun the regression to see what the data really looks like.


# In[10]:

data = compareTwoFeatures('salary', 'bonus', data_dict, "SALARY versus BONUS cleansed of outliers")


# ## Task: Data exploration
# - Get descriptive statistics
# 
# 

# In[11]:

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




# In[12]:

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


# In[13]:

import pandas as pd


df = pd.DataFrame({'salary': salary, 'bonus': bonus, 'exercisedStockOptions': exerStockOptions, 
                   'restrictedStock': restrictedStock})

ax = df['salary'].plot()
ax.yaxis.tick_right()

_= plt.xlabel('datapoint value')
_= plt.title('Salary')

_= plt.legend(loc='upper center', shadow=True, fontsize='medium')
plt.show()

df['salary'].describe()


# In[14]:

ax2 = df['exercisedStockOptions'].plot()
ax2.yaxis.tick_right()

_= plt.xlabel('datapoint value')
_= plt.title('Exercised Stocks')

_= plt.legend(loc='upper center', shadow=True, fontsize='medium')

plt.show()

df['exercisedStockOptions'].describe()


# In[15]:

ax2 = df['restrictedStock'].plot()
ax2.yaxis.tick_right()

_= plt.xlabel('datapoint value')
_= plt.title('Restricted Stock')

_= plt.legend(loc='upper center', shadow=True, fontsize='medium')

plt.show()

df['restrictedStock'].describe()


# In[16]:

ax2 = df['bonus'].plot()
ax2.yaxis.tick_right()

_= plt.xlabel('datapoint value')
_= plt.title('Bonus')

_= plt.legend(loc='upper center', shadow=True, fontsize='medium')

plt.show()

df['bonus'].describe()


# In[17]:

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
           
    


# In[18]:

f1, f2 = 'salary','exercised_stock_options'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus EXERCISED STOCK OPTIONS")


# In[19]:

treshold = 8000000
#printLatex(f1, f2, data_dict, treshold)


# In[20]:

f1, f2 = 'salary','restricted_stock'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus RESTRICTED STOCK")



# In[21]:

treshold = 3000000
#printLatex(f1, f2, data_dict, treshold)


# In[22]:

f1, f2 = 'salary','bonus'
data = compareTwoFeatures(f1, f2, data_dict, "SALARY versus BONUS")


# In[23]:

treshold = 4000000
#printLatex(f1, f2, data_dict, treshold)


# In[24]:

data_dict['LAVORATO JOHN J']


# ### Engineered Feature
# - #### Fraction of messages to and from POI

# ## Task 3: Create new feature(s)
# - features_list is a list of strings, each of which is a feature name.
# - The first feature must be "poi".
# - Store to `my_dataset` for easy export below.

# In[25]:

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
    


# In[26]:

f1, f2 = 'bonus','exercised_stock_options'
data = compareTwoFeatures(f1, f2, data_dict, 'Bonus v Exercised Stock Options')


# Extract features and labels from dataset

# In[27]:

feature_list = ['poi', 'bonus', 'exercised_stock_options', 'restricted_stock']
# 'fraction_from_poi','fraction_to_poi',
# 'from_poi_to_this_person','from_this_person_to_poi'
#  'salary'
my_dataset = data_dict

data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "The features we are using to train our model are as follows:"
print feature_list


# In[28]:

import pandas as pd

data = pd.DataFrame({'poi':labels})
ax = data.plot.hist()
ax.yaxis.tick_right()

_= plt.xlabel('datapoint value')
_= plt.title('Histogram of Person of Interest')
_= plt.legend(loc='upper center', shadow=True, fontsize='medium')
_= plt.axis([0, 1, 0, 200])
plt.show()


# # Pick and Tune an Algorithm
# ## Task 4: Try a variety of classifiers
# - Please name your classifier clf for easy export below.
# - Note that if you want to do PCA or other multi-stage operations, you'll need to use Pipelines. For more info: http://scikit-learn.org/stable/modules/pipeline.html
# 
# # Validate and Evaluate
# ## Task 5: Tune your classifier
# - Achieve better than .3 precision and recall. Using our testing script. Check the `tester.py` script in the final project folder for details on the evaluation method, especially the test_classifier function. Because of the small size of the dataset, the script uses `stratified shuffle split cross validation`. For more info: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 

# In[29]:

feature_list


# In[30]:

from sklearn.cross_validation import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from time import time


print '_'*20, 'RandomForestClassifier', '_'*20
print "Training the data"

clf = RandomForestClassifier(n_estimators=100) #, class_weight = {0: 0.1, 1: 0.9}
t0 = time()
scores = cross_val_score(clf, features, labels)
print("done in %0.3fs" % (time() - t0))
print "Scores for the prediction: ", scores, "\n"        
    
#print"Running learner against tester script\n"    
#t0 = time()    
#test_classifier(clf, my_dataset, feature_list, folds = 1000)
#print("done in %0.3fs" % (time() - t0))


# In[31]:

from sklearn.ensemble import ExtraTreesClassifier

print '_'*20, 'ExtraTreesClassifier', '_'*20
print "Training the data"

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, 
                           min_samples_split=1, random_state=0)

t0 = time()
scores = cross_val_score(clf, features, labels)
print("done in %0.3fs" % (time() - t0))
print "Scores for the prediction: ", scores, "\n"        
    
#print"Running learner against tester script\n" 

#t0 = time()    
#test_classifier(clf, my_dataset, feature_list, folds = 1000)
#print("done in %0.3fs" % (time() - t0))


# In[32]:

from sklearn.linear_model import LogisticRegression

print '_'*20, 'LogisticRegression', '_'*20
print "Training the data"

clf = LogisticRegression(penalty='l1', solver='liblinear')

t0 = time()
scores = cross_val_score(clf, features, labels)
print("done in %0.3fs" % (time() - t0))
print "Scores for the prediction: ", scores, "\n"        
    
#print"Running learner against tester script\n"        
    
    
#t0 = time()    
#test_classifier(clf, my_dataset, feature_list, folds = 1000)
#print("done in %0.3fs" % (time() - t0))


# In[33]:



from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

print '_'*20, 'Tuning LogisticRegression', '_'*20
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



# In[36]:

clf =  grid.best_estimator_
print "final learner"
print clf


#print"Running the tuned learner against the tester script\n"        
        
#t0 = time()    
#test_classifier(clf, my_dataset, feature_list, folds = 1000)
#print("done in %0.3fs" % (time() - t0))


# Running the tuned learner against the tester script
# 
# LogisticRegression(C=0.125, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.01,
#           verbose=0, warm_start=False)
#           
# 	Accuracy: 0.86431	Precision: 0.72957	Recall: 0.18750	F1: 0.29833	F2: 0.22023
# 	Total predictions: 13000	True positives:  375	False positives:  139	False negatives: 1625	True negatives: 10861
# 
# done in 0.694s

# ## Task 6: Export solution
# Dump your classifier, dataset, and features_list so anyone can check your results. You do not need to change anything below, but make sure that the version of `poi_id.py` that you submit can be run on its own and generates the necessary .pkl files for validating your results.

# In[37]:

dump_classifier_and_data(clf, my_dataset, feature_list)


# In[ ]:



