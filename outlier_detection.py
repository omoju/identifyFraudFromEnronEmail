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

    
    