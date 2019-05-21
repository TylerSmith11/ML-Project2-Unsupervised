#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:57:31 2018

@author: tyler
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def catFeatureDict(catArray,catAttrNames):
    catDict = []
    for row in catArray:
        dictRow = {}
        for col in range(len(row)):
            dictRow[catAttrNames[col]] = row[col]
        catDict.append(dictRow)
    
    return catDict


data = np.genfromtxt('trainData.csv',delimiter=',',dtype=None,
                         encoding=None)
trainFeatures = data[::2]
testFeatures = data[1::2]
    
   
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

rng = np.random.RandomState(42)

clf = IsolationForest(max_samples=100, random_state=rng, contamination='auto')

clf.fit(trainAll)

clf.predict(testAll)
