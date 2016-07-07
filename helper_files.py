import sys
dataPath = '../../IntroToMachineLearning/'
sys.path.append(dataPath+'tools/')
sys.path.append(dataPath+'final_project/')

from feature_format import featureFormat
import matplotlib.pyplot as plt

def compareTwoFeatures(feature1, feature2, the_data_dict):
    """
    Compare two features in a scatter plot and returns the datapoints
   
    Keyword arguments:
    feature1 -- an (n x k) numpy array where n is 2 and k datapoints 
    feature2 -- an (n x k) numpy array where n is 2 and k datapoints
   
    """
    features_list =  [ feature1, feature2]
    data = featureFormat(the_data_dict, features_list, remove_any_zeroes=True)

    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( x,y )

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    _= plt.title('Comparing two features for outliers')
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
        print "{:20}{:12}${:<12,.2f}{:12}${:<12,.2f}".format(item[0], feature1+' is ', 
                                                             item[1], ' '+feature2+' is ', item[2])
    


    
    