import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import njit

kernel = np.array([[-1],
                   [0],
                   [1]])

kernel2 = np.array([[-1, 0, 1]])
window_size = 8

@njit()
def cal_histogram(angles, mags):
    SizeX = int((angles[0].shape[0]) / window_size)
    SizeY = int((angles[0].shape[1]) / window_size)

    #histogram_list = np.array([[[0.0 for _ in range(9)] for _ in range(SizeX)] for _ in range(SizeY)])
    histogram_list = np.zeros((SizeY, SizeX, 9))
    for x in range(0, SizeX):
        for y in range(0, SizeY):
            r = x * window_size
            c = y * window_size
            # window = gray[r:r + window_size, c:c + window_size]
            # cv.imshow("Image:" + str(r) + " : " + str(c), window)

            for i in range(0, window_size):
                for j in range(0, window_size):
                    working_angle = 0.0
                    working_mag = 0.0
                    for k in range(len(angles)):
                        # This is for bins spaced over 0◦–180◦, i.e. the ‘sign’ of the gradient is ignored.
                        if mags[k][r + i][c + j] > working_mag:
                            working_angle = angles[k][r + i][c + j]
                            working_mag = mags[k][r + i][c + j]

                    if working_angle > 180:
                        working_angle -= 180

                    # 180 grader og ikke 360
                    bin = int(working_angle / 20.0)
                    per_for_next_upper = ((bin + 1) * 20 - working_angle) / 20
                    per_for_next_bottom = (working_angle - bin * 20) / 20

                    if bin == 8:
                        histogram_list[y, x][bin] += working_mag * per_for_next_bottom
                        histogram_list[y, x][0] += working_mag * per_for_next_upper
                    else:
                        histogram_list[y, x][bin] += working_mag * per_for_next_bottom
                        histogram_list[y, x][bin + 1] += working_mag * per_for_next_upper

    return histogram_list


def calculate_Hog_OPENCV(frame, notUsed=None):
    winSize = (64,128)
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
    locations = ((0,0),)
    hist = hog.compute(frame,winStride,padding,locations)
    return hist.ravel(), None


def calculate_Hog(frame, draw_arrows = False):
    channels = cv.split(frame)

    mags = []
    angles = []
    for channel in channels:
        gy = (np.float32(cv.filter2D(channel, -1, kernel)) / 255.0)
        gx = (np.float32(cv.filter2D(channel, -1, kernel2)) / 255.0)
        mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
        mags.append(mag)
        angles.append(angle)

    histogram_list = cal_histogram(angles, mags)

    # print(histogram_list.shape)
    # fig, axs = plt.subplots(histogram_list.shape[1], histogram_list.shape[0])
    # for i, var_name in enumerate(histogram_list):
    #     for j, var_name2 in enumerate(histogram_list[i]):
    #         axs[j, i].hist(histogram_list[i, j], bins=9)

    feature_vector = []
    e = 0.0000001
    for x in range(0, histogram_list.shape[0] - 1):
        row_x = []
        for y in range(0, histogram_list.shape[1] - 1):
            new_hist = np.array([]).astype(np.float32)
            for x_new in range(0, 2):
                for y_new in range(0, 2):
                    new_hist = np.append(new_hist, np.array(histogram_list[x + x_new, y + y_new]))
            # L2-Hys normalize then clip then normalize.
            divide_value = np.sqrt(np.linalg.norm(new_hist) ** 2 + e)
            new_table = [min(x, 0.2) for x in (new_hist / divide_value)]
            divide_value2 = np.sqrt(np.linalg.norm(new_table) ** 2 + e)
            feature_vector.append(new_table / divide_value2)
        #feature_vector.append(row_x)

    #print(f'Number of HOG features = {len(feature_vector) * len(feature_vector[0]) * len(feature_vector[0][0])}')

    if (draw_arrows):
        for x in range(0, histogram_list.shape[0]):
            for y in range(0, histogram_list.shape[1]):
                center_x = x * window_size + int(window_size / 2)
                center_y = y * window_size + int(window_size / 2)
                for i, value in enumerate(histogram_list[x, y]):
                    end_x = int(center_x + value * np.sin((i * 20) * np.pi / 180) * 20)
                    end_y = int(center_y + value * np.cos((i * 20) * np.pi / 180) * 20)
                    cv.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255))

    return np.reshape(feature_vector, len(feature_vector)*len(feature_vector[0])), frame


if __name__ == "__main__":
    cap = cv.VideoCapture(r"Test.mp4")
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        histrograms, frame = calculate_Hog(frame, True)

        plt.figure()
        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGRA2RGB))
        plt.show()
