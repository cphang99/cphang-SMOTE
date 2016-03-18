import unittest
import data_set
import numpy
import smote_transform
import ada_syn

class test_smote(unittest.TestCase):
    def setUp(self):
        self.datasetPhoneme = data_set.dataSet('./datasets/phoneme.csv')
        self.datasetGlass = data_set.dataSet('./datasets/glass5.csv')
        self.smote = smote_transform.smoteTransform(randomState = 255)
        self.adaSyn = ada_syn.adaSyn(randomState=255)
    
    def test_phonemeValidate(self):
        auc = self.smote.validate(self.datasetPhoneme.data, self.datasetPhoneme.labels, smotePercentages = [100])
        self.assertEqual([0.87058040140989978], auc)
        
    def test_phonemeUnderValidate(self):
        auc = self.smote.undersampleValidate(self.datasetPhoneme.data, self.datasetPhoneme.labels)
        self.assertEqual([0.86915696291175371], auc)
    
    def test_adaSyn(self):
        auc = self.adaSyn.validate(self.datasetGlass.data, self.datasetGlass.labels, adaSynBeta=[0.5], toStandardise=False, toShuffle=True, kfolds=10)
        self.assertEqual([0.71002005012531333], auc)
     
if __name__ == '__main__':
    unittest.main()
    
    