import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import cProfile
from pstats import Stats

from calc_hog import calculate_Hog
from annotation_parser import parseDataset


if __name__ == '__main__':
    with cProfile.Profile() as pr:

        imgs, neg_image_filenames = parseDataset()

        numObjects = 0
        for img in imgs:
            numObjects += len(img.objects)
        print('Total images: ', len(imgs), ' with a total of ', numObjects, ' objects')

        positives = []

        for i, image in enumerate(imgs):
            if i == 10:
                break
            print('image ', i)
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

                feature, frame = calculate_Hog(imgCrop)
                positives.append(feature)

                #plt.figure()
                #plt.imshow(imgCrop)            
            #plt.show()

        stats = Stats(pr)
        stats.sort_stats('time')
        stats.print_stats()