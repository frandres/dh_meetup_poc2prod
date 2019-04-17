import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 


class FlowerRatioCalculator( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self,parameters = None):
        self.parameters = parameters 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        X = np.array(X)
        X = np.append(X, (X[:,0]/X[:,1]).reshape(-1,1), axis=1)
        X = np.append(X, (X[:,2]/X[:,3]).reshape(-1,1), axis=1)
        return X