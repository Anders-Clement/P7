
import cv2 as cv
import pickle
from sklearn import svm
import matplotlib.pyplot as plt

from calc_hog import calculate_Hog

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    clf = pickle.load(open('model.pickle', 'rb'))

    imgs = ['hog2.jpeg']
    # for im in imgs:
    #     frame_raw = cv.imread(im)
    while True:
        ret, frame_raw = cap.read()
        frame_raw = cv.rotate(frame_raw, cv.ROTATE_90_COUNTERCLOCKWISE)
        if not ret:
            break

        for i in range(4):
            fig, ax = plt.subplots(2,1)
            frame = cv.resize(frame_raw, (0,0), fx=(1-i*0.2), fy=(1-i*0.2))
            y_len,x_len,_= frame.shape
            step_x = 8
            step_y = 24
            scalex = int(x_len / step_x)
            scaley = int(y_len / step_y)
                
            for y in range(scaley):
                if (y)*step_y + 96 > frame.shape[0]:
                    continue
                for x in range(scalex):
                    if ((x)*step_x + 32) > frame.shape[1]:
                        continue
                    cropped_image=frame[(y*step_y):((y)*step_y + 96),
                                        (x*step_x):((x)*step_x + 32)]
            
                    ax[1].imshow(cropped_image)
                    feature, notUsedVariableBecauseItIsUseless = calculate_Hog(cropped_image)
                    pred = clf.predict_proba([feature])
                    if pred[0,0] > pred[0,1]:
                        if pred[0,0] > 0.9:
                            print(pred[0])
                            cv.rectangle(frame, (x*step_x, y*step_y), (x*step_x + 32, y*step_y + 96), (255,0,0))
                        # ax[0].imshow(frame)
                        # plt.draw() 
                        # plt.waitforbuttonpress()
                        # plt.pause(0.00001)
            ax[0].imshow(frame)
            plt.show()
