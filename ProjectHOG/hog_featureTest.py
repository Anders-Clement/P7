
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def calculate_Hog(frame, draw_arrows = False):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(frame,winStride,padding,locations)

    return np.reshape(hist, len(hist)), frame
    #return np.reshape(feature_vector, len(feature_vector)*len(feature_vector[0])), frame


if __name__ == "__main__":
    cap = cv.VideoCapture(r"/home/hax/syncfolder/P7/P7Group/ProjectHOG/ProjectHOG/Test.mp4")
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        histrograms, frame = calculate_Hog(frame, True)

        plt.figure()
        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGRA2RGB))
        plt.show()
