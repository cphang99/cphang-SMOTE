"""
A series of helper methods for smoteTransform.py, testPlatform.py primarily, but also for gradientBoostTest.py and adaSyn.py

"""

import numpy
import math
import sklearn.metrics as mets
from sklearn.preprocessing import StandardScaler

def sampler(dataFile):
    ids = dataFile[:,0]
    data = dataFile[:,1:-1] #last but one feature in 2D array
    labels = dataFile[:,-1] #last 'feature' in 2D array i.e. the label
    #print(data, labels)
    
    return ids, data, labels

def getCharacteristics(data):
    return data.shape[0], data.shape[1]

def samplerNoID(dataSet):
    data = dataSet[:,:-1] #last but one feature in 2D array
    labels = dataSet[:,-1] #last 'feature' in 2D array i.e. the label
    
    return data, labels

def combineTestSets(data1, labels1, data2, labels2):
    totData = numpy.concatenate((data1, data2), axis=0)
    totLabels = numpy.concatenate((labels1, labels2), axis=0)
    
    return totData, totLabels

def sampleFromIndices(data, labels, train_indices, val_indices, standardise=False):
    trData = data[train_indices]
    trLabel = labels[train_indices]
    vaData = data[val_indices]
    vaLabel = labels[val_indices]
    
    if not standardise:
        return trData, trLabel, vaData, vaLabel
    else:
        vaScaledData = StandardScaler().fit_transform(vaData)
        trScaledData = StandardScaler().fit_transform(trData)
        return trScaledData, trLabel, vaScaledData, vaLabel

def addRow(m, r):
    return numpy.vstack((m, r))

def fallout(label, preds):
    cm = mets.confusion_matrix(label, preds)
    #This occurs iff in a 1x1 matrix occurs where all values within label and preds are the same, and the classes in label and preds
    #are the same, or if the classes in label and preds are different.
    #Hence test the first instance and return the boolean outcome as 0 or 1
    if (getCharacteristics(cm)[0] and getCharacteristics(cm)[1]) == 1:
        return int(label[0] != preds[0])
    
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[0,1]
    if fp == 0:
        return 0
    else:
        return float(fp) / float(fp + tn)

def calculateAUC(scores):
    #print(scores)
    scores = addRow(scores, [1, 1])
    return mets.auc(scores[:,1], scores[:,0], reorder=True)

def confidence95(score):
    #print(len(score))
    return 1.96 * (numpy.std(score) / math.sqrt(len(score)))

def saveDataSets(data, labels, filename):
    numpy.savetxt(filename + '.csv', numpy.concatenate((data, numpy.c_[labels]), axis=1), delimiter = ',')
