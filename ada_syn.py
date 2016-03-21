"""
This module implements the adaSyn algorithm
Note that this implementation is only suitable for datasets which have binary classification.

"""
import numpy
from sklearn import neighbors
from dataParsing import *
import smote_transform
from sklearn import cross_validation

class adaSyn(smote_transform.smoteTransform):
    """Performs the ADASYN transform on unbalanced data sets.

    Attributes:
        k: number of nearest neighbors. Default = 5
        underSamplePercentage: How much to undersample the majority class. Default = 15 (%)
        minorityLabel: the class of the minority label. Default =1
        randomState: Sets a seed for any random generator functions within the smoteTransform object for testing. 
                        Default = None
        beta: The adaSyn beta parameter.                
    """    
    def __init__(self, k=5, oversampleParameter=1, underSamplePercentage = 15, minorityLabel =1, randomState=None):
        smote_transform.smoteTransform.__init__(self, k=k, 
                                                underSamplePercentage=underSamplePercentage, 
                                                minorityLabel=minorityLabel, randomState=randomState)
        self.beta = oversampleParameter
        self.dthreshold = 0.75
        self.densityclf = None
    
    
    def adaSynAdd(self, data, labels):
        """Determines the number of synthetic samples to generate for each minority example, and generates the synthetic datapoints.

        Args:
            data: examples with all their features
            labels: classlabels for all corresponding examples.
            
            returns: a set of synthetic datapoints

        """
        r = {}
        g = {}
        rnorm = {}
        rsum = 0
        self.fit(data, labels)
        self.densityclf = neighbors.KNeighborsClassifier(n_neighbors=self.k)    
        self.densityclf.fit(data, labels)
        
        #Note that this is an alternative approach for extracting the minority examples
        #in the *same* order as described in smoteTransform.fit()
        for index in xrange(0, len(data)):
            if labels[index] == abs(1 - self.minorityLabel):
                continue
            
            nrpoints = self.densityclf.kneighbors(data[index,:], return_distance=False)
            nrpoints = numpy.setdiff1d(nrpoints, [index])
            if self.minorityLabel == 1:
                num_majority = self.k - numpy.count_nonzero(labels[nrpoints])
            else:
                num_majority = numpy.count_nonzero(data[nrpoints])
                
            r[index] = float(num_majority) / float(self.k)
            assert(r[index] >= 0)
        
        
        for k, v in r.viewitems(): 
            #print(k,v)
            rsum += v
        for k, v in r.viewitems():
            rnorm[k] = r[k] / rsum
            
        rnormsum = 0
        for k, v in rnorm.viewitems(): rnormsum += v
        #print(rnormsum)
        
        #m = mj + ml, -> if mj = m - ml, mj - ml = m - 2(ml)
        #where len(data) = m and len(r) = mj
        
        #Number of synthetic samples to generate
        G = float(len(data) - len(r) - len(r)) * float(self.beta)
        index = 0
        numNewPoints = 0
        #Convert normalised density distribution values to the number of values
        #to generate for each minority sample.
        for k, v in rnorm.viewitems():
            g[index] = int(round(rnorm[k] * G))
            numNewPoints += g[index]
            index += 1
        
        #print(numNewPoints)
        #print(self.minorityData)
        #Use this information to the smoteTransform transfer function.
        #for k, v in g.viewitems(): print(k,v)
        #len(g)
        #len(data[labels == 1])
        assert len(g) == len(data[labels == 1]), "length of g ({0}) is different from num_minority ({1})".format(len(g), len(data[labels == 1]))
        return self.transform(numRepeatArray = g)

    def getProcessedData(self, data, labels):
        """Conveinence method. From a given dataset with corresponding labels, will return
        a new data set which has had its majority class undersampled (optional) and its minority class oversampled (optional).
        This is an overriden method of the base smoteTransform class that implements the adaSyn algorithm instead
            
        Returns: A dataset consisting of an oversampled minority class and an undersampled majority class
                 as a tuple (data, labels)

        """
        if self.underSamplePercentage != 0:
            data, labels = self.underSample(data, labels)
        if self.beta != 0:    
            synData, synLabels = self.adaSynAdd(data, labels)
            if synData is not None:
                data, labels = combineTestSets(data, labels, synData, synLabels)
        return data, labels 
    

#Used to test the adaSyn class, takes a CSV file, puts it into numpy objects and sends it to the adaSyn class.         
def main():        
    csvFileName = './datasets/glass5.csv'
    dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
    data, labels = samplerNoID(dataSet)
    #processedData, processedLabels = adaSyn().getProcessedData(data, labels, underSamplePercentage=50, adaSynBeta=0.5)
    #saveDataSets(processedData, processedLabels, "CombinedPlayerSetAda")
    adaSyn(randomState=255).validate(data, labels, adaSynBeta=[0.5], toStandardise=False, toShuffle=True, kfolds=10)
    
if __name__ == '__main__':
    #import doctest
    #doctest.testmod()
    main()
        
        
        
        