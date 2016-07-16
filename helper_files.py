import sys
dataPath = '../../IntroToMachineLearning/'
sys.path.append(dataPath+'tools/')
sys.path.append(dataPath+'final_project/')

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt



#Stylistic Options for plots
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)
        
def makeData(dataset, feature_list, folds = 1000):
    """Make and return dataset prepared for training.

    Keyword arguments:
    dataset --- dict of dict
    feature_list --- list of strings
    folds --- int
    
    """
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    
    for train_idx, test_idx in cv: 
        
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
    return features_train, features_test, labels_train, labels_test

def getImportantFeatures(rf, data_dict, n=1, topNFeatures=5):
    """Return topNFeatures as found by a Random Forest Classifier.

    Keyword arguments:
    rf --- Random Forest Classifier
    dataset --- dict of dict
    n --- int
    topNFeatures --- int
    
    """
    import operator
    featuresSortedByScore = list()
    
    feature_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                'bonus', 'restricted_stock_deferred', 'deferred_income', 
                'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
                'long_term_incentive', 
                'restricted_stock', 'director_fees',
                'to_messages','from_poi_to_this_person', 'from_messages', 
                'from_this_person_to_poi', 'shared_receipt_with_poi'
               ]

    names = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
        'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
        'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
         'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 
                'from_this_person_to_poi', 'shared_receipt_with_poi']
    
    data = featureFormat(data_dict, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    for i in range(n):
        rf.fit(features, labels)
        featuresSortedByScore.append(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))

    myTopFeature = {}
    for i in range (len(featuresSortedByScore)):
        for j in range (topNFeatures):
            if featuresSortedByScore[i][j][1] not in myTopFeature.keys():
                myTopFeature[featuresSortedByScore[i][j][1]] = 1
            else:
                myTopFeature[featuresSortedByScore[i][j][1]] += 1


    return sorted(myTopFeature.items(), key=operator.itemgetter(1), reverse=True)[:topNFeatures]

def printDataTableAtThreshold(feature1, feature2, the_data_dict, threshold, prtLaTex):
    """Print output table for feature values > threshold.

    Keyword arguments:
    feature1 --- string
    feature2 --- sring
    the_data_dict --- dict of dict
    threshold --- int
    prtLaTex --- int
    
    """
    def getKey(item):
        return item[2]

    temp = []
    for key, value in the_data_dict.iteritems():
        if (value[feature1] != "NaN") and (value[feature2] != "NaN" and value[feature2] > threshold):
            temp.append(( key, value[feature1], value[feature2]))
            
    ### print out in ascending order of feature2    
    temp = sorted(temp, key=getKey)
    
    ## print output formatted ready for LaTex
    if prtLaTex:
        print "{:20}{:3}{:14}{:3}{:12}{:3}".format("Name".upper(), '&', feature1.upper(), '&', feature2.upper(), '\\\\')
    
        for item in temp:    
            print "{:20}{:3}\${:<14,.2f}{:3}{:12,}{:3}".format(item[0], '&', item[1], '&', item[2], '\\\\')
    else:
        print "{:20}{:14}{:12}".format("Name".upper(), feature1.upper(), feature2.upper())
    
        for item in temp:    
            print "{:20}${:<14,.2f}{:12,}".format(item[0], item[1], item[2])
        

        
def printClassifierOutputTable(results, prntLatex):
    """Print output table for classifier scores.

    Keyword arguments:
    results -- array of dict
    prntLatex -- 1 for print optimized for latex, 0 for regular printing
    
    """
    
    print "{:^67}".format('Metrics')
    print "{:25}{:12}{:10}{:10}".format("", 'precision', 'recall', 'F1')
    print "-"*63

    if prntLatex:
        for i in range(0, len(results)):
            print "{:25}{:3}{:<12.3f}{:3}{:<10.3f}{:3}{:<10.3f}{:3}".format(results[i]['clfname'],'&', results[i]['precision'], 
                                                     '&', results[i]['recall'], '&', results[i]['f1'], '\\\\')
    else:
        for i in range(0, len(results)):
            print "{:25}{:<12.3f}{:<10.3f}{:<10.3f}".format(results[i]['clfname'], results[i]['precision'], 
                                                 results[i]['recall'], results[i]['f1'])
        
                 
def validateClf(clf, dataset, feature_list, folds = 1000):
    """Validate classifier and return dict with the results.

    Keyword arguments:
    clf --- Classifier
    dataset --- dict of dict
    feature_list --- list of strings
    folds --- int
    
    """
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



def compareTwoFeatures(feature1, feature2, df):
    """
    Compare two features in a scatter plot and returns the datapoints
   
    Keyword arguments:
    feature1 -- string 
    feature2 -- string
    df --- pandas dataframe
   
    """
  
    plt.scatter( df[feature1], df[feature2], s=80, marker = 'o' ,color=tableau20[0], alpha = 0.5)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    _= plt.title(feature1.upper()+' versus '+feature2.upper(), y=1.08)
    plt.show() 

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
        
        Keyword arguments:
        poi_messages -- string 
        all_messages -- string
        
   """
    
    fraction = 0
    
    if poi_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    


    return fraction

def findPersonBasedOnTwoFeatures(feature1, feature2, the_data_dict, treshold=1):
    """
    Compare two features based on a treshold
   
    Keyword arguments:
    feature1 -- a string representing a data feature like 'salary'
    feature2 -- a string representing a data feature like 'salary'
    the_data_dict --- dict of dict
    treshold -- an int
   
    """
    
    def getKey(item):
        return item[2]

    temp = []
    for key, value in the_data_dict.iteritems():
        if (value[feature1] != "NaN") and (value[feature2] != "NaN" and value[feature2] > treshold):
            temp.append(( key, value[feature1], value[feature2]))
            
    ### print out in ascending order of feature2    
    temp = sorted(temp, key=getKey)
    for item in temp:    
        print "{:20}{:12}${:<12,.2f}{:12}{:<12}".format(item[0], feature1+' is ', 
                                                             item[1], ' '+feature2+' is ', item[2])
    


    
    