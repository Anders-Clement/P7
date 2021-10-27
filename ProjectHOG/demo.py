
import cv2 as cv
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

from calc_hog import calculate_Hog_OPENCV as calculate_Hog
from training import getTrainedModel

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    model = "SVM"

    if model == "SVM":
        clf = pickle.load(open('model_linear_svm.pickle', 'rb'))
    elif model == "NN":
        clf = getTrainedModel()
    else:
        print("wrong model name supplied")
        exit(-1)

    imgs = ['hog2.jpeg', 'hog3.jpeg']
    for im in imgs:
        frame_raw = cv.imread(im)
        frame_raw = cv.cvtColor(frame_raw, cv.COLOR_BGR2RGB)
    # while True:
    #     ret, frame_raw = cap.read()
        #frame_raw = cv.rotate(frame_raw, cv.ROTATE_90_COUNTERCLOCKWISE)
        # if not ret:
        #     break
        rectsToDraw = []
        for i in range(3):
            fig, ax = plt.subplots(2,1)
            frame = cv.resize(frame_raw, (0,0), fx=(1-(i+1)*0.2), fy=(1-(i+1)*0.2))
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
                    if model == 'SVM':
                        pred = clf.predict([feature])
                    
                    elif model == 'NN':
                        f = np.array([feature])
                        pred = clf.predict(f)

                    #if pred[0,1] > 0.9:
                    if pred == '1':
                        print(pred[0])
                        rectsToDraw.append([frame, (x*step_x, y*step_y), (x*step_x + 64, y*step_y + 128), (255,0,0)])
                    # ax[0].imshow(frame)
                    # plt.draw() 
                    # plt.waitforbuttonpress()
                    # plt.pause(0.00001)
            for rect in rectsToDraw:
                cv.rectangle(rect[0], rect[1], rect[2], rect[3])
            ax[0].imshow(frame)
            plt.show()
