
import cv2 as cv
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os

from calc_hog import calculate_Hog_OPENCV as calculate_Hog
from feature_extractor import get_pos_neg_samples_from_pickle
from training import getTrainedModel
from non_maximal_suppression import non_max_suppression


def sliding_window_demo(clf):
    imgs = []
    path = os.path.join('.', 'INRIAPerson/Test/pos/')
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if str(file).endswith('.png'):
                imgs.append(os.path.join(root, file))
                
    fig, ax = plt.subplots(1,2)
    for im in imgs:
        frame_raw = cv.imread(im)
        frame_raw = cv.cvtColor(frame_raw, cv.COLOR_BGR2RGB)
        rectsToDraw = []
        for i in range(4):
            NNwindows = []
            NNrects = []
            fx=(1-(i+1)*0.2)
            fy=(1-(i+1)*0.2)
            frame = cv.resize(frame_raw, (0,0), fx=fx, fy=fy)
            y_len,x_len,_= frame.shape
            step_x = 8
            step_y = 16
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
            
                    #ax[1].imshow(cropped_image)
                    feature, notUsedVariableBecauseItIsUseless = calculate_Hog(cropped_image)
                    hog = cv.HOGDescriptor()
                    if model == 'nonlinear_SVM':
                        pred = clf.predict_proba([feature])
                        if pred[0,1] > 0.9:
                            rectsToDraw.append([x*step_x, y*step_y, x*step_x + 64, y*step_y + 128, pred[0,1]])
                    elif model == 'linear_SVM':
                        pred = clf.predict([feature])
                        if pred == '1':
                            rectsToDraw.append([x*step_x /fx, y*step_y/fy, (x*step_x + 64)/fx, (y*step_y + 128)/fy, 1.0])
                    elif model == 'NN':
                        NNwindows.append(feature)
                        NNrects.append([x*step_x, y*step_y, x*step_x + 64, y*step_y + 128])
                        #pred = clf.predict(f)

                    # ax[0].imshow(frame)
                    # plt.draw() 
                    # plt.waitforbuttonpress()
                    # plt.pause(0.00001)
            if model == 'NN':
                # for win in NNwindows:
                #     w = win[0]
                #     rect = win[1]
                #     f = np.array([np.array(w)])
                #     pred = clf.predict(f)
                #     print(pred)
                windows = np.array(NNwindows)
                #tensor = tf.convert_to_tensor(windows)
                pred = clf.predict(windows)

                for i, p in enumerate(pred):
                    if p[0] > 0.9:
                        NNrects[i].append(p[0])
                        rectsToDraw.append(NNrects[i])



        
        frame_copy = deepcopy(frame_raw)
        for rect in rectsToDraw:
            cv.rectangle(frame_copy, (int(rect[0]),int(rect[1])), (int(rect[2]), int(rect[3])), (255,0,0))
        ax[0].imshow(frame_copy)
        rectsToDraw = non_max_suppression(np.array(rectsToDraw), 0.5)
        for rect in rectsToDraw:
            cv.rectangle(frame_raw, (int(rect[0]),int(rect[1])), (int(rect[2]), int(rect[3])), (255,0,0))
        ax[1].imshow(frame_raw)
        plt.draw()
        plt.waitforbuttonpress()


def test_window_demo(clf):
    print('extracting samples from images...')
    pos_samples, neg_samples = get_pos_neg_samples_from_pickle(fileNamePos='pos_test_samples.pickle', 
                                                                fileNameNeg='neg_test_samples.pickle',
                                                                dataFolder='INRIAPerson/Test/')

    print('classifying...')
    TP = 0
    FP = 0
    FN = 0
    for pos in pos_samples:
        pred = clf.predict([pos])
        if pred == '1':
            TP += 1
        else:
            FN += 1
    
    for neg in neg_samples:
        pred = clf.predict([neg])
        if pred == '1':
            FP += 1

    print('TP: ', TP, ', FP: ', FP, ', FN: ', FN)
    precision = TP/(TP + FP)
    recall = TP/(TP+FN)
    print('Precision: ', precision, ', recall: ', recall)


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    model = "linear_SVM"

    if model == "linear_SVM":
        clf = pickle.load(open('model_linear_svm.pickle', 'rb'))
    elif model == 'nonlinear_SVM':
        clf = pickle.load(open('model_nonlinear_svm.pickle', 'rb'))
    elif model == "NN":
        clf = getTrainedModel()
    else:
        print("wrong model name supplied")
        exit(-1)

    test_window_demo(clf)
    
