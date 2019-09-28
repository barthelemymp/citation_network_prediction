
"""
This is the prediction file

"""

###############
# IMPORTATION #
###############

import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import csv
import nltk
import xgboost as xgb
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

##################################
# LOAD THE DATA IN FORMAT NUMPY  #
##################################

training_features=np.load("data_npy/training_features.npy")
training_labels=np.load("data_npy/training_labels.npy")
testing_features=np.load("data_npy/testing_features.npy")

##########################
# REBALANCE THE DATASET  #
##########################

''' We keep only 80% of the 0-labelled elements'''

total_1=np.sum([training_labels])

to_keep=[]

i=0
cpt=0 # Number of 0 we keep

while cpt<0.8*(training_labels.shape[0]-total_1):
	to_keep+=[i]
	if training_labels[i]==0:
		cpt+=1
	i+=1

# We have 80% of the 0-labelled elements, we add the remaining 1-labelled elements
for j in range(i,len(training_labels)):
	if training_labels[j]==1:
		to_keep+=[j]

training_features=np.take(training_features,to_keep,axis=0)
training_labels=np.take(training_labels,to_keep,axis=0)

#################
# TRAINING STEP #
#################

# Preprocess the training_features 
training_features=preprocessing.scale(training_features)

# The best result had been obtained with a RF
classifier = RandomForestClassifier(n_jobs=1, n_estimators=700, criterion="gini", min_samples_split=10,
                                    min_samples_leaf=2, max_features="sqrt", max_depth=10)

# Training step
print("initiate training")
classifier.fit(training_features, training_labels)
print("end")

###################
# PREDICTION STEP #
###################

#Scale the testing features
testing_features=preprocessing.scale(testing_features)

#Predict
predictions = list(classifier.predict(testing_features))


# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_features)), predictions_SVM) #Maybe a pb with len(testing_features)

with open("predictions.csv","wb") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(['id', 'category'])
    for row in predictions_SVM:
        csv_out.writerow(row)












