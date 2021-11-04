import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle

import cProfile
from pstats import Stats

from calc_hog import calculate_Hog_OPENCV as calculate_Hog
from annotation_parser import parseDataset


def get_pos_neg_samples(saveToFile=False, dataFolder='INRIAPerson/Train/'):
    imgs, neg_image_filenames = parseDataset(dataFolder)
    numObjects = 0
    for img in imgs:
        numObjects += len(img.objects)
    positive_samples = []
    for i, image in enumerate(imgs):
        img = cv.imread(image.fileName, 0)

        for j, obj in enumerate(image.objects):
            padding = 0
            yMin = int(obj.yMin - obj.yMin/6)
            if yMin < 0:
                yMin = 0
            yMax = int(obj.yMax + obj.yMax/6) 
            if yMax > image.imageShape[1]:
                yMax = image.imageShape[1]
            xMin = int(obj.xMin - obj.xMin/2) 
            if xMin < 0:
                xMin = 0
            xMax = int(obj.xMax + obj.xMin/2)
            if xMax > image.imageShape[0]:
                xMax = image.imageShape[0]
            imgCrop = img[yMin : yMax, xMin : xMax]
            imgCrop = cv.resize(imgCrop, (64,128))
            feature, frame = calculate_Hog(imgCrop)
            positive_samples.append(feature)
            feature, frame = calculate_Hog(cv.flip(imgCrop, 1))
            positive_samples.append(feature)

    
    negative_samples = []

    for n in neg_image_filenames:
        neg_img = cv.imread(n)
        if neg_img is None:
            continue
        for i in range(10):
            x = np.random.randint(0, neg_img.shape[1] - 64)
            y = np.random.randint(0, neg_img.shape[0] - 128)
            imgCrop = neg_img[y : y + 128, x : x + 64]
            feature, frame = calculate_Hog(imgCrop)
            negative_samples.append(feature)

    if(saveToFile):
        print('done calculating features, saving to file...')
        with open('positive_features.txt', 'w') as output:
            for image in imgs:
                output.write(','.join(str(e) for e in image.feature_vector))
                output.write('\n')

        print('wrote ', len(imgs), ' features to positive_features.txt')

    return positive_samples, negative_samples
    
def saveSamples(pos_samples, neg_samples, fileNamePos='pos_samples.pickle', fileNameNeg='neg_samples.pickle'):
    pickle.dump(pos_samples, open(fileNamePos, 'wb'))
    pickle.dump(neg_samples, open(fileNameNeg, 'wb'))

def get_pos_neg_samples_from_pickle(fileNamePos='pos_samples.pickle', fileNameNeg='neg_samples.pickle', dataFolder='INRIAPerson/Train/'):
    try:
        pos_samples = pickle.load(open(fileNamePos, 'rb'))        
        neg_samples = pickle.load(open(fileNameNeg, 'rb'))
    except Exception as e:
        print('Could not read samples from pickle, extracting from images instead')
        pos_samples, neg_samples = get_pos_neg_samples(dataFolder=dataFolder)
        saveSamples(pos_samples, neg_samples,fileNamePos=fileNamePos, fileNameNeg=fileNameNeg)

    return pos_samples, neg_samples


if __name__ == '__main__':
    positives, negatives = get_pos_neg_samples()
    saveSamples(positives, negatives)
    print('calculated ', len(positives), ' positive train samples, and ', len(negatives), ' negative train samples')
    print('dumped samples to pickle')
    positives, negatives = get_pos_neg_samples(dataFolder='INRIAPerson/Test/')
    saveSamples(positives, negatives, 'pos_test_samples.pickle', 'neg_test_samples.pickle')
    print('calculated ', len(positives), ' positive test samples, and ', len(negatives), ' negative test samples')
    print('dumped samples to pickle')


