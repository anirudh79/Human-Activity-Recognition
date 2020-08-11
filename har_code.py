import sys

import scipy
import numpy as np
import matplotlib
import pandas
import sklearn

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


print("Test branch");

f_y_train = open("/home/anirudh/Downloads/UCI_HAR_Dataset/train/y_train", 'r')
f_x_train = open("/home/anirudh/Downloads/UCI_HAR_Dataset/train/x_train", 'r')

x_train_array = np.fromfile(f_x_train, dtype=float, count=-1, sep=' ')
y_train_array = np.fromfile(f_y_train, dtype=int, count=-1, sep='\n')

# Given the total number of features are 561, so the array is reshaped
x_train_array = x_train_array.reshape(-1, 561)

print('The total number of cases used to train are {}\n' .format(y_train_array.size))

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []

print('The name of classifier used and its corresponding performance on the test set is:\n')

for name, model in models:
    kfold = cross_validation.KFold(n=1838, n_folds=10, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, x_train_array, y_train_array, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('Linear Discriminant Analysis is used to fit the training set')

lda = LinearDiscriminantAnalysis()
lda.fit(x_train_array, y_train_array)

f_y_test = open("/home/anirudh/Downloads/UCI_HAR_Dataset/test/y_test.txt", 'r')
f_x_test = open("/home/anirudh/Downloads/UCI_HAR_Dataset/test/X_test.txt", 'r')

x_test_array = np.fromfile(f_x_test, dtype=float, count=-1, sep=' ')
y_test_array = np.fromfile(f_y_test, dtype=int, count=-1, sep='\n')

x_test_array = x_test_array.reshape(-1, 561)


predictions = lda.predict(x_test_array)
print('The accuracy of the classifier on the test set is {}'.format(accuracy_score(y_test_array, predictions)))
