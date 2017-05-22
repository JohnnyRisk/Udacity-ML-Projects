#%%
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
t0 = time()
gaus = clf.fit(features_train,labels_train)
t1 = time()
gaus.predict(features_test)
t2 = time()
print("This is the score from score {}".format(gaus.score(features_test, labels_test)))
print("training time is {}s".format(round(t1-t0,3)))
print("test time is {}s".format(round(t2-t1,3)))
print("This is the score from accuracy {}".format(accuracy_score(labels_test, gaus.predict(features_test))))

#########################################################


