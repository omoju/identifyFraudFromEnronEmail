import sys
dataPath = '../../IntroToMachineLearning/'
sys.path.append(dataPath+'tools/')
sys.path.append(dataPath+'final_project/')

from feature_format import featureFormat
import matplotlib.pyplot as plt

def compareTwoFeatures(feature1, feature2, data_dict):
    """
    Compare two features in a scatter plot and returns the datapoints
   
    Keyword arguments:
    feature1 -- an (n x k) numpy array where n is 2 and k datapoints 
    feature2 -- an (n x k) numpy array where n is 2 and k datapoints
   
    """
    features_list =  [ feature1, feature2]
    data = featureFormat(data_dict, features_list, remove_any_zeroes=True)

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


    
    