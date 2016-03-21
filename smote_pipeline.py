import sklearn.base
import smote_transform
import dataParsing

class smotePipeline:
    def __init__(self, steps):
        self.steps = steps
        self.finalEstimator = None
    
    def fit(self, data, labels):
        estimator = None
        for step in self.steps[:-1]:
            process = step()
            try:
                data, labels = process.getProcessedData(data, labels)
            except:
                data =  process.fit_transform(data, labels)
        else:
            process = step()
            self.finalEstimator = process.fit(data, labels)
        
        return True
     
        
    
