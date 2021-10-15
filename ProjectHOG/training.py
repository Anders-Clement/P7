
from operator import ne
from numpy.lib.npyio import save
from sklearn import svm
from feature_extractor import get_pos_neg_samples, get_pos_neg_samples_from_pickle, saveSamples
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
import pickle


if __name__ == '__main__':
    positives, negatives = get_pos_neg_samples()
    saveSamples(positives,negatives)
    #positives, negatives = get_pos_neg_samples_from_pickle()
    data = np.array(positives + negatives, dtype='float32')
    labels = np.zeros(len(data),dtype='str')
    labels[:len(positives)] = 'human'
    labels[len(positives):] = 'not a human'
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)

    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_pred, y_test))
    
    #pickle.dump(clf, open('model.pickle', 'wb'))
    