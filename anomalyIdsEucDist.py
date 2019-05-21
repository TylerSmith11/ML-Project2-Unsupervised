# -*- coding: utf-8 -*-
"""
Sample anomaly detector by using Euclidean distance from mean over all
training data.

@author: Kevin S. Xu
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor


def predictAnomalies(trainFeatures,testFeatures):
    nAttr = len(trainFeatures.dtype)
    nCatAttr = 3
    nRealAttr = nAttr - nCatAttr
    nTrainSamples = np.size(trainFeatures)
    nTestSamples = np.size(testFeatures)
    
    # Get list of names of categorical attributes
    catAttrNames = list(trainFeatures.dtype.names[-nCatAttr:])
    
    # Convert categorical features to binary using 1-of-K representation    
    trainCat = trainFeatures[catAttrNames]
    trainCatDict = catFeatureDict(trainCat,catAttrNames)
    dv = DictVectorizer()
    trainCatEncoded = dv.fit_transform(trainCatDict).toarray()
    testCat = testFeatures[catAttrNames]
    testCatDict = catFeatureDict(testCat,catAttrNames)
    testCatEncoded = dv.transform(testCatDict).toarray()    
    
    # Extract real features and convert all to float type
    trainReal = np.zeros((nTrainSamples,nRealAttr))
    testReal = np.zeros((nTestSamples,nRealAttr))
    for attr in range(nRealAttr):
        trainReal[:,attr] = trainFeatures['f' + str(attr)].astype(float)
        testReal[:,attr] = testFeatures['f' + str(attr)].astype(float)
    
    # Combine real features and encoded categorical features (now all of type
    # float)
    trainAll = np.c_[trainReal,trainCatEncoded]
    testAll = np.c_[testReal,testCatEncoded]
    
    # Simple anomaly detector--compute distance of each test sample from mean
    # over all training samples
    
    #LOF "Large values corespond to inliers, abs makes lower more normal (need to normalize data)
    outlierFactor = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination="auto")
    outlierFactor.fit(trainAll)
    outlierScore = outlierFactor.score_samples(testAll)
    outlierScore = np.abs(outlierScore)
    outlierScore = (outlierScore - min(outlierScore)) / (max(outlierScore) - min(outlierScore))
    
    
    #gives 0 to 1 results
    SVMfunction = svm.OneClassSVM(kernel = "rbf", gamma="auto")
    SVMfunction.fit(trainAll)
    SVMScore = SVMfunction.score_samples(testAll)
    SVMScore = np.abs(SVMScore)
    SVMScore = (SVMScore - min(SVMScore)) / (max(SVMScore) - min(SVMScore))

    
    #isoforest the more negative the more abnormal np.abs to abso value more positive is more abnormal now
    isolateForest = IsolationForest(contamination="auto",behaviour="new")
    isolateForest.fit(trainAll)
    isoScore = isolateForest.score_samples(testAll)
    isoScore = np.abs(isoScore)
    isoScore = (isoScore - min(isoScore)) / (max(isoScore) - min(isoScore))

    
    AverageScore = (outlierScore + SVMScore + isoScore) / 3
    
    return AverageScore

"""
Convert structured array of categorical variables (represented as byte
strings) to a list of dictionaries with values as decoded strings
"""
def catFeatureDict(catArray,catAttrNames):
    catDict = []
    for row in catArray:
        dictRow = {}
        for col in range(len(row)):
            dictRow[catAttrNames[col]] = row[col]
        catDict.append(dictRow)
    
    return catDict
    
if __name__ == "__main__":
    data = np.genfromtxt('trainData.csv',delimiter=',',dtype=None,
                         encoding=None)
    trainData = data[::2]
    testData = data[1::2]
    anomScores = predictAnomalies(trainData,testData)
    print(anomScores)
    f = open("Scores.txt", "w")
    i = 1
    for score in anomScores:
        f.write(str(i))
        f.write(")")
        f.write(str(score))
        f.write("\n")
        i = i +2
    f.close
    