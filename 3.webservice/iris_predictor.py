from sklearn.externals import joblib
import os

class IrisPredictor():
    def __init__(self,path):
        self.__model = joblib.load(path)
        
    def predict(self,data):
        return self.__model.predict(data)