import sys
dataPath = '../../IntroToMachineLearning/'
sys.path.append(dataPath+'tools/')
sys.path.append(dataPath+'final_project/')

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt


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
        
            
    print "\n\n"
    print "{:25}{:12}{:10}{:10}{:10}".format("", "TP", 'FP', 'TN', 'FN')
    print "-"*63

    for i in range(0, len(results)):
        print "{:25}{:<12}{:<10}{:<10}{:<10}".format(results[i]['clfname'], results[i]['true_positives'], 
                                                     results[i]['false_positives'], results[i]['true_negatives'],
                                                            results[i]['false_negatives'])
        
def validateClf(clf, dataset, feature_list, folds = 1000):
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



def compareTwoFeatures(feature1, feature2, the_data_dict, title):
    """
    Compare two features in a scatter plot and returns the datapoints
   
    Keyword arguments:
    feature1 -- an (n x k) numpy array where n is 2 and k datapoints 
    feature2 -- an (n x k) numpy array where n is 2 and k datapoints
   
    """
    
    #Stylistic Options for plots
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
    
    
    features_list =  [ feature1, feature2]
    data = featureFormat(the_data_dict, features_list, remove_any_zeroes=True)

    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( y, x, s=80, marker = 'o' ,color=tableau20[0], alpha = 0.5)

    plt.xlabel(feature2)
    plt.ylabel(feature1)
     #plt.grid(True)
    _= plt.title(title)
    plt.show() 
    
    return data

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
        
   """
    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    
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
    


    
    