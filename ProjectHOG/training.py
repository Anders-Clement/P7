from sklearn import svm
from feature_extractor import get_pos_neg_samples, get_pos_neg_samples_from_pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pickle

from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers


def getTrainingData():
    positives, negatives = get_pos_neg_samples_from_pickle()
    data = np.array(positives + negatives, dtype='float32')
    labels = np.zeros(len(data),dtype='str')
    labels[:len(positives)] = 1
    labels[len(positives):] = 0
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)
    return (X_train, X_test, y_train, y_test)


def getTrainedModel():
    X_train, X_test, y_train, y_test = getTrainingData()
    # print("x_train shape:", X_train.shape)
    # print(X_train.shape[0], "train samples")
    # print(X_test.shape[0], "test samples")

    # # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    """## Build the model"""
    model = keras.Sequential(
        [
            keras.Input(shape=(X_train.shape[1])),
            layers.Dense(10, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    )

    model.summary()

    """## Train the model"""
    batch_size = 2000
    epochs = 25

    #opt = optimizers.gradient_descent_v2.SGD(momentum=0.1)
    opt = optimizers.adam_v2.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #keras.metrics.Precision(), keras.metrics.Recall(),

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model


if __name__ == '__main__':
      
    # get NN model
    model = getTrainedModel()
    """## Evaluate the trained model"""
    X_train, X_test, y_train, y_test = getTrainingData()
    #reorder data to fit Keras
    y_test = keras.utils.to_categorical(y_test)
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    pred = model.predict(X_test)

    #calculate and print metrics for NN
    TP = 0
    FP = 0
    FN = 0
    for i, p in enumerate(pred):
        if y_test[i,1]:
            if p[1] > p[0]:
                TP += 1
            else:
                FN += 1
        else:
            if p[1] > p[0]:
                FP += 1

    print('TP: ', TP, ', FP: ', FP, ', FN: ', FN)
    precision = TP/(TP + FP)
    recall = TP/(TP+FN)
    print('Precision: ', precision, ', recall: ', recall)
    
    # fit linear SVM, predict and print metrics
    X_train, X_test, y_train, y_test = getTrainingData()
    clf = svm.LinearSVC(max_iter=10000, C=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("linear SVM classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("linear SVM confusion matrix:")    
    print(metrics.confusion_matrix(y_pred, y_test))
    pickle.dump(clf, open('model_linear_svm.pickle', 'wb'))

    # fit non-linear SVM, predict and print metrics
    X_train, X_test, y_train, y_test = getTrainingData()
    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("non linear SVM classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("non linear SVM confusion matrix:")
    print(metrics.confusion_matrix(y_pred, y_test))
    pickle.dump(clf, open('model_nonlinear_svm.pickle', 'wb'))
    