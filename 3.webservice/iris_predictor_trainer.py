import os
import yaml
import git
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib
from custom_transformer import FlowerRatioCalculator


class IrisPredictorTrainer():
    def __init__(self,data_path):
        self.data_path = data_path
        self.loadDataset(data_path)
        
    def loadDataset(self,path):
        with open(path+"data_config.yaml", 'r') as stream:
            try:
                self.database_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        data = pd.read_csv(path+self.database_config["filename"],header=None)
        
        X,y = data.iloc[:,0:4],data.iloc[:,4]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
    def trainModel(self,hiperparam_pipeline):
        
        pipeline = Pipeline(steps=[("flower_ratio_calculator",FlowerRatioCalculator()),
                                   ("scaler",StandardScaler()), 
                                   ("predictor",SVC())])
        
        self.hiperparam_pipeline = hiperparam_pipeline
        
        self.model = GridSearchCV(estimator=pipeline,
                                       param_grid=hiperparam_pipeline,
                                       scoring="accuracy",
                                       cv=10,
                                       n_jobs=4 
                                      )
        
        self.model.fit(self.X_train,self.y_train)
        
        print("Best estimator params", self.model.best_params_)
        print("Best estimator cv accuracy",self.model.best_score_)
        
        self.test_score = self.model.score(self.X_test,self.y_test)
        
        print("Best estimator test accuracy",self.test_score)
        
    def storeModel(self,path,tag=""):
        if not os.path.exists(path):
            os.makedirs(path)
        
        model_config = {"hyper_parameter_grid": self.hiperparam_pipeline,
                        "best_hyper_parameter": self.model.best_params_,
                        "repository_tag": self.getRepoTag(),
                        "pipeline_steps": [step[0] for step in self.model.best_estimator_.steps],
                        "cv-score": str(self.model.best_score_),
                        "test-score": str(self.test_score),
                        "database_config":self.database_config,
                        "tag":tag}
        
        with open(path+'config.yml', 'w') as outfile:
            yaml.dump(model_config, outfile, default_flow_style=False)
    
        joblib.dump(self.model.best_estimator_, path+'model.pkl')
        
    def getRepoTag(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
