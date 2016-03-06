"""
This module implements the SMOTE algorithm as described in https://www.jair.org/media/953/live-953-2037-jair.pdf.
Note that this implementation is only suitable for datasets which have binary classification.

"""

import numpy
import scipy
import random
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier as treeClf
from dataParsing import *
from sklearn import cross_validation
import sklearn.metrics as mets
import sklearn.utils as utils

class smoteTransform:
    """Performs the SMOTE transform on unbalanced data sets.

    Uses fit and transform methods to conform with general API of scikit-learn
    Returns a set of synthetic datapoints. Would then require combination with the original dataset.

    Attributes:
        k: number of nearest neighbors. Default = 5
        smote: the oversampling SMOTE percentage. Default = 200
        underSamplePercentage: How much to undersample the majority class. Default = 15 (%)
        minorityLabel: the class of the minority label. Default =1
        randomState: Sets a seed for any random generator functions within the smoteTransform object for testing. 
                        Default = None
    """
    def __init__(self, k=5, smote=200, underSamplePercentage = 15, minorityLabel =1, randomState=None):
        """Initialisation of the smoteTransform object.

        Args:
            k: number of nearest neighbors. Default = 5
            smote: the oversampling SMOTE percentage. Default = 200
            underSamplePercentage: How much to undersample the majority class. Default = 15 (%)
            minorityLabel: the class of the minority label. Default =1
            randomState: Sets a seed for any random generator functions within the smoteTransform object for testing. 
                         Default = None
            
        Returns: a smoteTransform object

        """
        self.k = k + 1
        self.smote = smote
        self.minorityLabel = minorityLabel
        self.randomState = randomState
        self.underSamplePercentage = underSamplePercentage
        self.clf = None
        self.SynData = None
        self.SynLabels = None
        self.minorityExamples = None
        
        self.minorityData = None #Note this refers to original data, but *only* of the minority class.
        self.minorityLabels = None #Note this refers to original data, but *only* of the minority class.
        

    def fit(self, data, labels):
        """Fits the minority data to a knn classifier from a class

        Args:
            data: examples with all their features
            labels: classlabels for all corresponding examples.
            
            returns: a smoteTransform object with a fitted data model.

        """
        self.minorityExamples, self.minorityData, self.minorityLabels = self._getMinorityClass(data, labels)

        #use a knn classifier to identify the nearest neighbors we require
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=self.k)    
        self.clf.fit(self.minorityData, self.minorityLabels)
        
        return self
    
    def transform(self, adaSyn=False, g = None):
        """Generates synthetic samples according to the SMOTE algorithm

        returns: Synthetic data points with their requisite minority class label as a tuple (data, labels).

        """            
        assert(self.minorityData != None and self.minorityLabels != None and self.minorityExamples != None)
        doOnce = False
        newPoints = []
        #Iterate through all of the minority examples.
        numExamples, numFeatures = getCharacteristics(self.minorityData) #get all the important characteristics of the data
        index = 0 #so we know which index we're at
        numSyntheticSamples = 0 #so we know how many synthetic samples have been created!
        for ex in self.minorityExamples:
            
            #see above, this gets the nearest points of the current example
            #in this context a nearest neighbor is also classed as itself, which we don't want. 
            #So get 6 points, and do a set difference to get the true 5 nearest neighbors
            #print((self.minorityData[index,:]))
            #We also need to deal with a corner case if the number of minority examples is less than k in knn
            if len(self.minorityExamples) < self.k:
                if not doOnce:
                    print("WARNING: Number of minor class examples < k in knn, " 
                        "returning all {0} members".format(len(self.minorityExamples)))
                    doOnce = True
                
                nearestPoints = self.clf.kneighbors(self.minorityData[index,:], 
                                                    n_neighbors=len(self.minorityExamples), return_distance=False)
            else:
                nearestPoints = self.clf.kneighbors(self.minorityData[index,:], return_distance=False)
            
            nearestPoints = numpy.setdiff1d(nearestPoints, [index])
            #print(numExamples)
            #print(nearestPoints)
            
            #pick a random point from these nearest points, repeat according to the smote percentage (with floor division)
            if adaSyn:
                assert(g != None)
                numRepeat = g[index]
            else:
                numRepeat = self.smote // 100
            for r in xrange(0,numRepeat):
                random.seed(self.randomState)
                nrpoint = random.choice(nearestPoints)
                assert(nrpoint < numExamples)
                
                #Picks the parameters for the synthetic point.
                params = []
                #For each feature:
                for d in xrange(0,numFeatures):
                    #take the difference between the data point in question and a random nearest one
                    #multiply it by rand(0,1) and add it to the data point.
                    #this is one feature value of a synthetic point.
                    param = ((self.minorityData[nrpoint,d] - self.minorityData[index,d]) * random.random()) + self.minorityData[index,d]
                    params.append(param)
                    
                #make it clear its a synthetic point by using 999 as the id, then use list comprehension to flatten the list
                clusteredPoint = [[999], params, [self.minorityLabel]]
                syntheticPoint = [elem for clus in clusteredPoint for elem in clus] 
                
                #add to list and increment the counter.
                newPoints.append(syntheticPoint)
                numSyntheticSamples += 1
            
            index += 1
        
        
        numpy.savetxt('synpoints.csv', numpy.c_[newPoints], delimiter= ',')
        
        newPointsNPArray = numpy.asarray(newPoints)
        newId, newData, newLabels = sampler(newPointsNPArray)
        return newData, newLabels


    def underSample(self, data, labels):
        """Undersamples the majority class in an imbalanced dataset.
        Args:
            data: examples with all their features
            labels: classlabels for all corresponding examples.
        
        Returns: a dataset with a set quantity of the majority class removed (undersampled) as a tuple of (data, labels).

        """
        random.seed(self.randomState)
        majorityExamples, minorityExamples = self._getClassSplit(data, labels)
        numMajorityExamples = 0
        if self.underSamplePercentage < 100:
            numMajorityExamples = int(float(len(majorityExamples)) * ((100-float(self.underSamplePercentage))/100))
        else:
            numMajorityExamples = len(minorityExamples) * 100/self.underSamplePercentage
            if numMajorityExamples == 0:
                numMajorityExamples = 1
        
        
        assert(numMajorityExamples > 0), str(self.underSamplePercentage)
        majorityExamplesIndices = random.sample(xrange(0,len(majorityExamples)), numMajorityExamples)
        majorData, majorLabels = samplerNoID(majorityExamples[majorityExamplesIndices])
        minorData, minorLabels = samplerNoID(minorityExamples)
        
        #print(len(majorData))
        #print(len(minorData))
        totData, totLabels = combineTestSets(majorData, majorLabels, minorData, minorLabels)
        
        return totData, totLabels
    

    def validate(self, data, labels, toStandardise=False, smotePercentages = None, toShuffle=False, kfolds=10):
        """Generates data-points (fp and tp) for generating a ROC curve through oversampling and undersampling of datasets
        See the original SMOTE paper for more details. All datapoints are the product of a 10-fold cross-validation.
        Original paper uses a C4.5 tree classifier. As we're using the scikit-learn library, we're using the CART algorithm
        instead, which is closely related.
        
        The smote percentages are as follows: 100, 200, 300, 400, 500
        The undersampling percentages are as follows: 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 
                                                      500, 600, 700, 800, 1000, 2000
                                                      
        For each smote percentage, the AUC is calculated and a csv file is saved corresponding to:
        smote%, undersample%, mean recall, mean fallout, 95% confidence (recall), 95% confidence (fallout)
        This file is saved as sample[smote%]auc=[AUC]_.csv
        Args:
            data: examples with all their features
            labels: classlabels for all corresponding examples.
            toStandardise: whether the datasets should be standardised (0 mean with unit variance) before fitting.
        
        >>> csvFileName = './datasets/phoneme.csv'
        >>> dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
        >>> data, labels = samplerNoID(dataSet)
        >>> smote = smoteTransform(randomState = 255)
        >>> smote.validate(data, labels, toStandardise=False, smotePercentages = [100])
        [[  0.00000000e+00   0.00000000e+00   7.92701288e-01   8.59985133e-02
            2.16956217e-02   7.65178918e-03]
         [  1.00000000e+02   1.00000000e+01   8.21095371e-01   1.10608309e-01
            1.87150305e-02   8.52243578e-03]
         [  1.00000000e+02   1.50000000e+01   8.28010859e-01   1.14282258e-01
            1.68657997e-02   5.16455238e-03]
         [  1.00000000e+02   2.50000000e+01   8.30418682e-01   1.17519395e-01
            2.54211142e-02   6.41673418e-03]
         [  1.00000000e+02   5.00000000e+01   8.60524004e-01   1.63941404e-01
            2.01031660e-02   1.11970736e-02]
         [  1.00000000e+02   7.50000000e+01   8.93129126e-01   2.34031882e-01
            2.04015059e-02   1.35582703e-02]
         [  1.00000000e+02   1.00000000e+02   8.60447911e-01   1.83000938e-01
            1.53169479e-02   1.40703712e-02]
         [  1.00000000e+02   1.25000000e+02   8.73333354e-01   2.03270958e-01
            1.56129049e-02   1.00728793e-02]
         [  1.00000000e+02   1.50000000e+02   8.80693877e-01   2.24381154e-01
            1.26417483e-02   1.05386373e-02]
         [  1.00000000e+02   1.75000000e+02   9.09395043e-01   2.35797524e-01
            1.56761029e-02   1.22288626e-02]
         [  1.00000000e+02   2.00000000e+02   9.05789727e-01   2.57232198e-01
            1.55591484e-02   1.19576636e-02]
         [  1.00000000e+02   3.00000000e+02   9.26321301e-01   2.99820813e-01
            1.26413492e-02   1.71827786e-02]
         [  1.00000000e+02   4.00000000e+02   9.44636370e-01   3.36573143e-01
            1.40436316e-02   2.37434116e-02]
         [  1.00000000e+02   5.00000000e+02   9.53258513e-01   3.63518690e-01
            1.44544332e-02   1.59074368e-02]
         [  1.00000000e+02   6.00000000e+02   9.53837342e-01   3.93044650e-01
            1.42175574e-02   1.75316626e-02]
         [  1.00000000e+02   7.00000000e+02   9.55234648e-01   4.06388475e-01
            7.66580273e-03   1.77307562e-02]
         [  1.00000000e+02   8.00000000e+02   9.62584962e-01   4.22494839e-01
            1.33841862e-02   2.26837287e-02]
         [  1.00000000e+02   1.00000000e+03   9.61396448e-01   4.49116790e-01
            1.05568599e-02   1.07949625e-02]
         [  1.00000000e+02   2.00000000e+03   9.78607059e-01   5.64731915e-01
            9.01519281e-03   2.96922772e-02]]
        0.87058040141
        [0.87058040140989978]
        """
        
        #Percentages used in the ROC paper, with 10-fold cross validation
        if not smotePercentages:
            smotePercentages = [100, 200, 300, 400, 500]
        assert(type(smotePercentages) is list)
        
        underSamplingPercentages = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
        kfold_iterator = cross_validation.KFold(len(data), shuffle=toShuffle, n_folds=kfolds, random_state=self.randomState)
        aucs = []
        #The initial datapoint is evaluation of the model without any over- or under-sampling.
        initialScore = None
        initialRecalls, initialfallouts = self.cvTree(data, labels, kfold_iterator, noDataChange = True,
                                                      toStandardise=toStandardise)
        initialScore = [0, 0, numpy.mean(initialRecalls), numpy.mean(initialfallouts), 
                        confidence95(initialRecalls), confidence95(initialfallouts)]
        #print(initialScore)
        
        #Then for each oversample percentage, do a series of undersampling of the majority class
        #This makes 1 ROC curve.
        for ovsample in smotePercentages:
            scores = initialScore
            for unsample in underSamplingPercentages:
                totRecall, totfallout = self.cvTree(data, labels, kfold_iterator, 
                                                    toStandardise=toStandardise, underAndOversample =True,
                                                    smotePercentage=ovsample, underSamplePercentage=unsample)
                    
                assert(scores != None)
                scores = addRow(scores, [ovsample, unsample, numpy.mean(totRecall), numpy.mean(totfallout), 
                                         confidence95(totRecall), confidence95(totfallout)])
                #print(ovsample, unsample, numpy.mean(totRecall), numpy.mean(totfallout), 
                      #confidence95(totRecall), confidence95(totfallout))
            
            auc = calculateAUC(scores[:,2:4])
            print(scores)
            numpy.savetxt('sample' + str(ovsample) + 'auc=' + str(auc)[:5] + '_.csv', scores, delimiter = ',')
            print(auc)
            aucs.append(auc)
        
        return aucs
    
    def undersampleValidate(self, data, labels, toStandardise=False, toShuffle=False, kfolds=10):
        """Generates data-points (fp and tp) for generating a ROC curve through oversampling and undersampling of datasets
        See the original SMOTE paper for more details. All datapoints are the product of a 10-fold cross-validation
        Original paper uses a C4.5 tree classifier. As we're using the scikit-learn library, we're using the CART algorithm
        instead, which is closely related.
        
        This function is similar to the validate method, but does not do any SMOTE oversampling. This is used as a comparison
        between only undersampling, and a combination of over- and under-sampling.
        
        The undersampling percentages are as follows: 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 
                                                      500, 600, 700, 800, 1000, 2000
                                                      
        The AUC is calculated and a csv file is saved corresponding to:
        undersample%, mean recall, mean fallout, 95% confidence (recall), 95% confidence (fallout)
        This file is saved as undersampleOnlyauc=[AUC]_.csv
        
        Args:
            data: examples with all their features
            labels: classlabels for all corresponding examples.
            toStandardise: whether the datasets should be standardised (0 mean with unit variance) before fitting.
        
        Returns: The auc from the resulting roc curve
        >>> csvFileName = './datasets/phoneme.csv'
        >>> dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
        >>> data, labels = samplerNoID(dataSet)
        >>> smote = smoteTransform(randomState = 255)
        >>> smote.undersampleValidate(data, labels, toStandardise=False)
        [[ 0.79270129  0.08599851]
         [ 0.78681043  0.09883799]
         [ 0.79016826  0.09998522]
         [ 0.80423374  0.11730889]
         [ 0.83255765  0.1435521 ]
         [ 0.87347219  0.21324935]
         [ 0.85003933  0.15736304]
         [ 0.85440382  0.19037023]
         [ 0.86097302  0.20513769]
         [ 0.86576104  0.217502  ]
         [ 0.89098296  0.24009012]
         [ 0.92390897  0.28186043]
         [ 0.93745969  0.31273222]
         [ 0.94237268  0.34739837]
         [ 0.9418215   0.3721151 ]
         [ 0.94033133  0.39831616]
         [ 0.9537111   0.41125992]
         [ 0.96623317  0.44349758]
         [ 0.98032728  0.53586075]]
        0.869156962912
        0.86915696291175371

        """        
        #under sampling percentages used in the SMOTE paper
        underSamplingPercentages = [0, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
        kfold_iterator = cross_validation.KFold(len(data), shuffle=toShuffle, n_folds=kfolds, random_state=self.randomState)
        
        scores = None
        doOnce = True
        for percentage in underSamplingPercentages:
                totRecall, totfallout = self.cvTree(data, labels, kfold_iterator, 
                                                    toStandardise=toStandardise,
                                                    undersampleOnly=True, underSamplePercentage=percentage)
                if doOnce:
                    doOnce = False
                    scores = [percentage, numpy.mean(totRecall), numpy.mean(totfallout), 
                              confidence95(totRecall), confidence95(totfallout)]
                else:
                    scores = addRow(scores, [percentage, numpy.mean(totRecall), numpy.mean(totfallout), 
                                             confidence95(totRecall), confidence95(totfallout)])
                assert(scores != None)
                #print(percentage, numpy.mean(totRecall), numpy.mean(totfallout), 
                      #confidence95(totRecall), confidence95(totfallout))
        
        print(scores[:,1:3])
        auc = calculateAUC(scores[:,1:3])
        numpy.savetxt('undersampleOnly' + 'auc=' + str(auc)[:5] + '_.csv', scores, delimiter = ',')
        print(auc)
        return auc
    
    def score(self, vaData, vaLabel, model):
        """Returns the recall and fallout score given the data, its corresponding labels and the model itself
        Args:
            vadata: validation data examples with all their features
            valabels: classlabels for all corresponding validation data examples.
            model: The model (which should already be trained with training data)
            
        Returns: The recall and fallout score as a tuple.

        """        
        vaPreds = model.predict(vaData)
        return mets.recall_score(vaLabel, vaPreds), fallout(vaLabel, vaPreds)
    
    def getProcessedData(self, data, labels, underSamplePercentage=None, oversample=False, smotePercentage=None):
        """Conveinence method. From a set of smote parameters and a given dataset with corresponding labels, will return
        a new data set which has had its majority class undersampled (optional) and its minority class oversampled (optional).
        The undersample and smote percentages can be set to different values from the original class variables.
        
        Args:
            vadata: validation set data
            vaLabel: Corresponding labels for the validation set data
            model: The model (which should already be trained with training data)
            
        Returns: A dataset consisting of an oversampled minority class and an undersampled majority class
                 as a tuple (data, labels)

        """ 
        if underSamplePercentage is not None:
            self.underSamplePercentage = underSamplePercentage
        if smotePercentage:
            self.smote = smotePercentage
        
        underSampledData, underSampledLabels = self.underSample(data, labels)
        if oversample:
            synData, synLabels = self.fit(data, labels).transform()
            totData, totLabels = combineTestSets(underSampledData, underSampledLabels, synData, synLabels)
            return totData, totLabels
        else:
            return underSampledData, underSampledLabels
    
    
    def cvTree(self, data, labels, kfold_iterator, 
               toStandardise=False, underAndOversample=False, noDataChange = False,
               undersampleOnly= False, smotePercentage=None, underSamplePercentage=None,
               adaSyn= False, adaSynBeta = None):
        """Performs cross validation.
        
        Args:
            data: test data
            labels: Corresponding labels for the test data
            kfold_iterator: The iterator used to generate the folds for the CV
            toStandardise: If the dataset is to be normalised
            underAndOversample: Should it be under and over-sampled?
            noDataChange: If CV should be performed on the original set
            undersampleOnly: Under sampling only
            smotePercentage: A set SMOTE percentage, otherwise just uses the class Attribute
            underSamplePercentage: see above
            adaSyn: Should the oversampling be done on using the AdaSyn alg?
            adaSynBeta: If adaSyn alg is to be done, set the beta parameter.
            
        Returns: A set of scores from the cross validation

        """ 
        totRecall = []
        totfallout = []
        for train_indice, val_indice in kfold_iterator:
            trData, trLabel, vaData, vaLabel = sampleFromIndices(data, labels, train_indice, val_indice, 
                                                                 standardise=toStandardise)
            if (numpy.all(trLabel == self.minorityLabel)) or (numpy.all(trLabel == abs(self.minorityLabel - 1))):
                if trLabel[0] == 1:
                    currentRecall = 0
                    currentFallout = 1
                else:
                    currentRecall = 1
                    currentFallout = 0
                print("WARNING: only one class present in label")
            elif noDataChange:
                totData, totLabels = trData, trLabel
            elif undersampleOnly:
                totData, totLabels = self.getProcessedData(trData, trLabel, underSamplePercentage=underSamplePercentage)
            elif underAndOversample:
                totData, totLabels = self.getProcessedData(trData, trLabel,
                                                           underSamplePercentage=underSamplePercentage,
                                                           oversample= True,
                                                           smotePercentage= smotePercentage)
            elif adaSyn:
                assert(adaSynBeta != None)
                totData, totLabels = self.getProcessedData(trData, trLabel,
                                                           underSamplePercentage=underSamplePercentage, adaSynBeta=adaSynBeta)
            else:
                raise AssertionError
                
            clf = treeClf(random_state = self.randomState)
            clf.fit(totData, totLabels)
                
            currentRecall, currentFallout = self.score(vaData, vaLabel, clf)
            
            totRecall.append(currentRecall)
            totfallout.append(currentFallout)
            #print(mets.recall_score(vaLabel, vaPreds), fallout(vaLabel, vaPreds))
        return totRecall, totfallout
    
    def _getMinorityClass(self, data, labels):
        """Extracts examples labelled with minority class from the overall data set.
        Args:
            data: datapoints corresponding to the minority label
            labels: the minority label
            
        returns: A dataset containing only examples labelled with the minority class as:
                 minorityExamples (data and labels combined), minorityData, minorityLabels

        """            
        #get the labels into a column vector
        clabels = numpy.reshape(labels, (labels.shape[0], 1))
        
        #concatenate with the data
        dataSet = numpy.concatenate((data, clabels), axis=1)
        
        #filter the dataset to include only the minority examples, and make a new dataset based on this.
        cond = (labels == self.minorityLabel)
        minorityExamples = dataSet[cond]
        #print(csv[cond])
        
        return minorityExamples, minorityExamples[:,:-1], minorityExamples[:,-1]

    def _getClassSplit(self, data, labels):
        """Splits a dataset into two separate numpy arrays corresponding to the majority class and the minority class.
        Args:
            data: datapoints corresponding to the minority label
            labels: the minority label
            
        returns: A dataset containing only examples labelled with the minority class as a tuple (majorityExamples, minorityExamples).

        """    
        majorityLabel = abs(self.minorityLabel - 1)
        
        #get the labels into a column vector
        clabels = numpy.reshape(labels, (labels.shape[0], 1))
        
        #concatenate with the data
        dataSet = numpy.concatenate((data, clabels), axis=1)
        
        majorityExamples = dataSet[labels == majorityLabel]
        minorityExamples = dataSet[labels == self.minorityLabel]
        
        return majorityExamples, minorityExamples
    
    


#Testing of the smoteTransform class
#Generates synthetic samples for further inspection.
def testPopulate():
    csvFileName = 'PlayersNumsOnly.csv'
    dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
    ids, data, labels = sampler(dataSet)
    smotePercentages = [200, 300, 400, 500]
    print(smotePercentages)
    for sp in smotePercentages:
        tr = smoteTransform(smote = sp)
        mindata, minlabels = tr.fit(data, labels).transform()
        numpy.savetxt('syn' + str(sp) + '.csv', numpy.concatenate((mindata, numpy.c_[minlabels]), axis=1), delimiter = ',')

#Tests the random undersampling of the majority class in the phoneme dataset.
#Generates synthetic samples for further inspection.        
def testUndersample():
    csvFileName = 'phoneme.csv'
    dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
    data, labels = samplerNoID(dataSet)
    tr = smoteTransform(smote = 200, underSamplePercentage = 500)
    synData, synLabels = tr.fit(data, labels).transform()
    data, labels = tr.underSample(data, labels)
    
    print(data, len(data))
    print(labels, len(labels))
    print(synData, len(synData))
    print(synLabels, len(synLabels))

#Generates data sets from re-sampling of a dataset into a CSV file, so that it can be displayed
#(such as by showtask() in mlotools)
def genDataSets(data, labels, filename):
    smote = smoteTransform(smote=500, underSamplePercentage=75)
    processedData, processedLabels = smote.getProcessedData(data, labels, oversample=True)
    saveDataSets(processedData, processedLabels, filename)
    

#Testing the smote function 200-500%
# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes Pima dataset
# http://sci2s.ugr.es/keel/dataset.php?cod=105 Phoneme dataset
def main():
    csvFileName = './datasets/glass5.csv'
    dataSet = numpy.genfromtxt(csvFileName, dtype=float, delimiter=",")
    data, labels = samplerNoID(dataSet)
    smote = smoteTransform(smote=400, underSamplePercentage=50)
    auc = smote.undersampleValidate(data, labels, toStandardise=False, toShuffle=True, kfolds=10)  
    aucs = smote.validate(data, labels, toStandardise=False, toShuffle=True, kfolds=10)  
    numpy.savetxt("auc_data", aucs, delimiter = ',')
    print("undersampling = 0 auc = {0}".format(auc))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #main()
