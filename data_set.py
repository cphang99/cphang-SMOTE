import numpy

class dataSet:
    def __init__(self, csvFile, hasID=False):
        self.csvFile = csvFile
        self.dataSet = numpy.genfromtxt(csvFile, dtype=float, delimiter=",")
        if not hasID:
            self.data, self.labels = self.samplerNoID()
        else:
            self.data, self.labels = self.sampler()
    
    def samplerNoID(self):
        data = self.dataSet[:,:-1] #last but one feature in 2D array
        labels = self.dataSet[:,-1] #last 'feature' in 2D array i.e. the label
        
        return data, labels
    
    def sampler(self):
        ids = self.dataSet[:,0]
        data = self.dataSet[:,1:-1] #last but one feature in 2D array
        labels = self.dataSet[:,-1] #last 'feature' in 2D array i.e. the label
        
        return ids, data, labels
    
    