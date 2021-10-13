import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle

import cProfile
from pstats import Stats

from calc_hog import calculate_Hog
from annotation_parser import parseDataset


def get_pos_neg_samples(saveToFile=False):
    imgs, neg_image_filenames = parseDataset()
    numObjects = 0
    for img in imgs:
        numObjects += len(img.objects)
    #print('Total images: ', len(imgs), ' with a total of ', numObjects, ' objects')
    positive_samples = []
    for i, image in enumerate(imgs):
        #print('image ', i)
        img = cv.imread(image.fileName, 0)
        #using matplotlib because opencv for some reason cannot show these images (it crashes)
        #plt.imshow(img)
        for obj in image.objects:
            padding = 16
            yMin = obj.yMin - padding
            if yMin < 0:
                yMin = 0
            yMax = obj.yMax + padding
            if yMax > image.imageShape[1]:
                yMax = image.imageShape[1]
            xMin = obj.xMin - padding
            if xMin < 0:
                xMin = 0
            xMax = obj.xMax + padding
            if xMax > image.imageShape[0]:
                xMax = image.imageShape[0]
            imgCrop = img[yMin : yMax, xMin : xMax]
            imgCrop = cv.resize(imgCrop, (32,96))
            #print('aspect: ', (yMax-yMin)/(xMax-xMin))
            feature, frame = calculate_Hog(imgCrop)
            positive_samples.append(feature)
            feature, frame = calculate_Hog(cv.flip(imgCrop, 1))
            positive_samples.append(feature)

        #     plt.figure()
        #     plt.imshow(cv.resize(imgCrop, (32,96)))
        #     plt.figure()
        #     plt.imshow(imgCrop)            
        # plt.show()

    
    negative_samples = []

    for n in neg_image_filenames:
        neg_img = cv.imread(n)
        if neg_img is None:
            print('skipping negative sample!')
            continue
        for i in range(10):
            x = np.random.randint(0, neg_img.shape[1] - 32)
            y = np.random.randint(0, neg_img.shape[0] - 96)
            imgCrop = neg_img[y : y + 96, x : x + 32]
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
    
def saveSamples(pos_samples, neg_samples):
    pickle.dump(pos_samples, open('pos_samples.pickle', 'wb'))
    pickle.dump(neg_samples, open('neg_samples.pickle', 'wb'))

def get_pos_neg_samples_from_pickle():
    pos_samples = pickle.load(open('pos_samples.pickle', 'rb'))
    neg_samples = pickle.load(open('neg_samples.pickle', 'rb'))
    return pos_samples, neg_samples


if __name__ == '__main__':
    positives, negatives = get_pos_neg_samples()
    saveSamples(positives, negatives)
    print('calculated ', len(positives), ' positive samples, and ', len(negatives), ' negative samples')
    print('dumped samples to pickle')


