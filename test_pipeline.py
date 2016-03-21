import unittest
import data_set
import numpy
import smote_transform
import ada_syn
import smote_pipeline
import sklearn.datasets
import sklearn.preprocessing
import sklearn.svm

class test_smote(unittest.TestCase):
    def setUp(self):
        self.steps = [smote_transform.smoteTransform, sklearn.preprocessing.MinMaxScaler, sklearn.svm.SVC]
        self.data, self.labels = sklearn.datasets.make_classification()
    
    def test_pipeline(self):
        self.assertTrue(smote_pipeline.smotePipeline(self.steps).fit(self.data, self.labels))