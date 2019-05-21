# -*- coding: utf-8 -*-
"""
Script to evaluate anomaly detection algorithm using AUC and true positive
rate at false positive rate <= 0.01

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as pl
from anomalyIdsEucDist import predictAnomalies
from sklearn.metrics import roc_auc_score,roc_curve

trainFeatures = np.genfromtxt('trainData.csv',delimiter=',',dtype=None,
                              encoding=None)
testData = np.genfromtxt('testData.csv',delimiter=',',dtype=None,
                         encoding=None)
testAttrNames = list(testData.dtype.names)
nTestSamples = np.size(testData)

# Separate labels and features from test data. Any example not from the normal
# class is treated as anomalous
testFeatures = testData[testAttrNames[:-1]]
testLabelsStr = testData[testAttrNames[-1]]
testLabels = np.where(testLabelsStr == 'normal',0,1)

anomScores = predictAnomalies(trainFeatures,testFeatures)
auc = roc_auc_score(testLabels,anomScores)
print('AUC score:', auc)

fpr,tpr,thres = roc_curve(testLabels,anomScores)
# True positive rate for highest false positive rate < 0.01
maxFprIndex = np.where(fpr<=0.01)[0][-1]
fprBelow = fpr[maxFprIndex]
fprAbove = fpr[maxFprIndex+1]
# Find TPR at exactly FPR = 0.01 by linear interpolation
tprBelow = tpr[maxFprIndex]
tprAbove = tpr[maxFprIndex+1]
tprAt = (tprAbove-tprBelow)/(fprAbove-fprBelow)*(0.01-fprBelow) + tprBelow
print('TPR at FPR =', fpr[maxFprIndex], ':', tpr[maxFprIndex])
print('TPR at FPR = 0.01:', tprAt)

pl.ion()
pl.plot(fpr,tpr)
pl.xlabel('False positive rate')
pl.ylabel('True positive rate')
pl.title('ROC curve for anomaly detector')
pl.show()
