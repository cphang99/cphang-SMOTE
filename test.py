import unittest
import data_set
import numpy
import smote_transform
import ada_syn
import smote_pipeline

import warnings
import traceback
import sys

class test_smote(unittest.TestCase):
    def setUp(self):
        self.randomState = 255
        self.datasetPhoneme = data_set.dataSet('./datasets/phoneme.csv')
        self.datasetGlass = data_set.dataSet('./datasets/glass5.csv')
        self.smote = smote_transform.smoteTransform(randomState = self.randomState)
        self.adaSyn = ada_syn.adaSyn(randomState=self.randomState)
        #warnings.showwarning = self.warn_with_traceback    
    
    ##From https://stackoverflow.com/questions/22373927/get-traceback-of-warnings
    #def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        #traceback.print_stack()

    
    #0.16 0.87058040140989978 initial
    #0.17 0.87149916311318743 new value with reshaping of the array (required with new API)
    def test_phonemeValidate(self):
        auc = smote_transform.validate(self.datasetPhoneme.data, self.datasetPhoneme.labels, overSamplingPercentages = [100], randomState=self.randomState)
        self.assertEqual([0.87149916311318743], auc)
    
    #0.16 0.86915696291175371 initial
    #0.17 0.87029500921696634 new value with reshaping of the array (required with new API)
    def test_phonemeUnderValidate(self):
        auc = smote_transform.undersampleValidate(self.datasetPhoneme.data, self.datasetPhoneme.labels, randomState=self.randomState)
        self.assertEqual([0.87029500921696634], auc)
    
    def test_adaSyn(self):
        auc = smote_transform.validate(self.datasetGlass.data, self.datasetGlass.labels, overSamplingPercentages=[0.5], 
                                       toStandardise=False, samplingMethodology=ada_syn.adaSyn, randomState = self.randomState, 
                                       toShuffle=True, kfolds=10)
        self.assertEqual([0.71002005012531333], auc)
     
if __name__ == '__main__':
    unittest.main()
    
    