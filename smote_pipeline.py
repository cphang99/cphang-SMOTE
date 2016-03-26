import sklearn.base
import smote_transform
import dataParsing

class smotePipeline(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, steps=None, randomState=None):
        if steps is None:
            steps = [smote_transform.smoteTransform]
        else:
            self.steps = steps
        self.finalEstimator = None
        self.randomState = randomState
    
    def fit(self, data, labels):
        estimator = None
        for step in self.steps[:-1]:
            process = step()
            try:
                data, labels = step(randomState=self.randomState).getProcessedData(data, labels)
            except:
                data =  process.fit_transform(data, labels)
        else:
            process = self.steps[-1]()
            process.fit(data, labels)
            self.finalEstimator = process
        
        return self
    
    def predict(self, data):
        try:
            return self.finalEstimator.predict(data)
        except:
            print("Unable to predict with final estimator")
     
        
    
