import numpy as np
from sklearn import svm, datasets, linear_model
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def get_data():
    return datasets.load_svmlight_file('genres3.libsvm')

def SVC():
    X, y = get_data()
    print('Support Vector Machine')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = svm.SVC(kernel='linear', C=.8)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    print("Confusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))
    print("Accuracy: " + str(classifier.score(X,y)) + '\n\n')

def SGD():
    X, y = get_data()
    print('Stochastic Gradient Descent')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = linear_model.SGDClassifier()
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    print("Confusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))
    print("Accuracy: " + str(classifier.score(X,y)) + '\n\n')

def NN():
    X, y = get_data()
    print('Nearest Neighbours')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = KNeighborsClassifier(n_neighbors=2)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    print("Confusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))
    print("Accuracy: " + str(classifier.score(X,y)) + '\n\n')

SVC()
SGD()
NN()
