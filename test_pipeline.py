import unittest
import data_set
import numpy
import smote_transform
import ada_syn
import smote_pipeline

import sklearn.datasets
import sklearn.preprocessing
import sklearn.svm
import sklearn.cross_validation

class test_smote(unittest.TestCase):
    def setUp(self):
        self.data, self.labels = sklearn.datasets.make_classification()
        self.trainingData, self.testingData, self.trainingLabels, self.testingLabels = sklearn.cross_validation.train_test_split(self.data, self.labels, random_state=255)
    
    #Tests manual and automatic construction of a list.
    def test_pipeline(self):
        steps = [smote_transform.smoteTransform, sklearn.preprocessing.MinMaxScaler, sklearn.svm.SVC]
        pipelinePredict = smote_pipeline.smotePipeline(steps, randomState=255).fit(self.trainingData, self.trainingLabels).predict(self.testingData)
        
        data, labels = smote_transform.smoteTransform(randomState=255).getProcessedData(self.trainingData, self.trainingLabels)
        data = sklearn.preprocessing.MinMaxScaler().fit_transform(data, labels)
        estimator = sklearn.svm.SVC().fit(data, labels)
        manualPredict = estimator.predict(self.testingData)
        
        self.assertTrue(numpy.array_equal(pipelinePredict, manualPredict))