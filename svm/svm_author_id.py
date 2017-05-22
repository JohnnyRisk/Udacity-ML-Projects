#%%
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
from sklearn.metrics import accuracy_score
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]
clf = svm.SVC(C = 10000, kernel ='rbf')
t0 = time()
clf.fit(features_train,labels_train)
t1 = time()
clf.predict(features_test)
t2 = time()
print("This is the score from score {}".format(clf.score(features_test, labels_test)))
print("training time is {}s".format(round(t1-t0,3)))
print("test time is {}s".format(round(t2-t1,3)))
print("This is the score from accuracy {}".format(accuracy_score(labels_test, clf.predict(features_test))))
#########################################################
#%%
pred = clf.predict(features_test)
print("number predicted as chris is {}".format(sum(pred)))
#%%
len(features_train[0])