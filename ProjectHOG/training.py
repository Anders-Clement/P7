from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pickle
import cv2 as cv

from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers

from feature_extractor import get_pos_neg_samples, get_pos_neg_samples_from_pickle
from annotation_parser import parseDataset
from calc_hog import calculate_Hog_OPENCV as calculate_Hog



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

def finalClassifier(clfPicklePath, X_train, y_train):
# Takes the fitted classifier and the data it used
# Using a sliding window we check all negative images for false-positives with the classifier
# All false-positives are then added to the dataset as negatives and we then train a new classifier on the extended dataset
# Returns newly trained/fitted calssifier
    old_x = X_train
    old_y = y_train
    classifier = pickle.load(open(clfPicklePath, 'rb'))
    positives, negatives = parseDataset('INRIAPerson/Train/')

    step_x = 8
    step_y = 16

    for n, neg in enumerate(negatives):
        print("Total_imgs/current_img: ",len(negatives),"/", n)
        if n > 200: break
        frame_raw = cv.imread(neg)
        if frame_raw is None:
            print("empty negative.png")
            continue
            
        for i in range(5):

            fx=(1-i*0.2)
            fy=(1-i*0.2)
            frame = cv.resize(frame_raw, (0,0), fx=fx, fy=fy)
            y_len,x_len,_= frame.shape
            scalex = int(x_len / step_x)
            scaley = int(y_len / step_y)
            
            for y in range(scaley):
                if (y)*step_y + 128 > frame.shape[0]:
                    continue
                for x in range(scalex):
                    if ((x)*step_x + 64) > frame.shape[1]:
                        continue
                    cropped_image=frame[(y*step_y):((y)*step_y + 128),
                                        (x*step_x):((x)*step_x + 64)]
                    feature, notUsedVariableBecauseItIsUseless = calculate_Hog(cropped_image)
                    if clfPicklePath == 'model_nonlinear_svm.pickle':
                        pred = classifier.predict_proba([feature])
                        if pred[0,1] > 0.75:
                            X_train = np.append(X_train, [feature], axis=0)
                            y_train = np.append(y_train,0)  
                    elif clfPicklePath == 'model_linear_svm.pickle':
                        pred = classifier.predict([feature])
                        if pred != '0':
                            X_train = np.append(X_train, [feature], axis=0)
                            y_train = np.append(y_train,0) 
 
    print("oldx & oldy",old_x.shape, old_y.shape)
    print("newx & newy",X_train.shape, y_train.shape)
    print("y diff (new pictures added)",y_train.shape[0]-old_y.shape[0])
    pickle.dump([X_train,y_train], open("N200__new_X_y_trainWithHardExamples.pickle", 'wb'))

    clfHardExamples = svm.LinearSVC(max_iter=10000, C=0.01)
    clfHardExamples.fit(X_train, y_train)
    
    return clfHardExamples
                      





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
        if y_test[i,0]:
            if p[0] > p[1]:
                TP += 1
            else:
                FN += 1
        else:
            FP += 1

    print('TP: ', TP, ', FP: ', FP, ', FN: ', FN)
    precision = TP/(TP + FP)
    recall = TP/(TP+FN)
    print('Precision: ', precision, ', recall: ', recall)

# fit linear SVM, predict and print metrics
    X_train, X_test, y_train, y_test = getTrainingData()
    clf = svm.LinearSVC(max_iter=10000, C=0.01)
    clf.fit(X_train, y_train)
    pickle.dump(clf, open('model_linear_svm.pickle', 'wb'))

    y_pred = clf.predict(X_test)
    print("linear SVM classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("linear SVM confusion matrix:")    
    print(metrics.confusion_matrix(y_pred, y_test))

# fit linear SVM again but with hard examples and predict and print metics
    hardExamples = True
    if hardExamples == True:
        new_clf = finalClassifier('model_linear_svm.pickle' ,X_train, y_train)
        pickle.dump(new_clf, open('new_model_linear_svm.pickle', 'wb'))
        new_y_pred = new_clf.predict(X_test)
        print("NEW linear SVM classification report:")
        print(metrics.classification_report(y_test, new_y_pred))
        print("NEW linear SVM confusion matrix:")
        print(metrics.confusion_matrix(new_y_pred, y_test))

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


