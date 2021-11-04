import os

class Object():
    def __init__(self, objLines) -> None:
        labelLine = objLines[3]
        #always 'PASperson', saving it anyways
        self.label = labelLine.split(':')[0].split('"')[1]
        #always 'UprightPerson', saving it anyways
        self.labelPose = labelLine.split(':')[1].split('"')[1]
        centerLine = objLines[4]
        self.centerX = int(centerLine.split(':')[1].split(',')[0][2:])
        self.centerY = int(centerLine.split(':')[1].split(',')[1][1:-2])
        bboxLine = objLines[5].split(':')
        xMinYMin = bboxLine[1].split('-')[0]
        xMaxYMax = bboxLine[1].split('-')[1]
        self.xMin = int(xMinYMin.split(',')[0][2:])
        self.yMin = int(xMinYMin.split(',')[1][1:-2])
        self.xMax = int(xMaxYMax.split(',')[0][2:])
        self.yMax = int(xMaxYMax.split(',')[1][1:-2])
        self.BboxShape = (self.xMax-self.xMin, self.yMax-self.yMin)


class Image():
    def __init__(self, fileName, imgShape,) -> None:
        self.fileName = fileName
        self.imageShape = imgShape
        self.objects = list()

    def addObject(self, object):
        self.objects.append(object)


def parseDataset(folder='INRIAPerson/Train/'):
    INRIA_FOLDER = os.path.join(os.getcwd(), 'INRIAPerson')
    TRAIN_FOLDER = os.path.join(os.getcwd(), folder)
    ANNOTATION_FOLDER = os.path.join(TRAIN_FOLDER, 'annotations')
    POS_FOLDER = os.path.join(TRAIN_FOLDER, 'pos')
    annotation_list = open(TRAIN_FOLDER + 'annotations.lst', 'r')
    pos_list = open(TRAIN_FOLDER + 'pos.lst', 'r')
    neg_list = open(TRAIN_FOLDER + 'neg.lst', 'r')

    neg_image_filenames = neg_list.readlines()

    neg_image_paths = []
    for neg in neg_image_filenames:
        if neg == '':
            continue
        neg_image_paths.append(os.path.join(INRIA_FOLDER, neg[:-1]))

    neg_list.close()


    imgs = []

    for annotation_file in annotation_list.readlines():
        annotation_file = annotation_file[:-1]
        pos_img_path = pos_list.readline()[:-2]

        if annotation_file is None or pos_img_path is None:
            print("threw up")
            exit(-1)
        with open(os.path.join(INRIA_FOLDER,annotation_file),'r', encoding='iso-8859-1') as annotation:
            lines = annotation.readlines()
            img_file = lines[2].split(':')[1][2:-2]
            size_line = lines[3].split(':')[1]
            sizes = size_line.split('x')
            x = int(sizes[0])
            y = int(sizes[1])
            c = int(sizes[2])
            imageShape = (x,y,c)

            img = Image(os.path.join(INRIA_FOLDER, img_file), imageShape)
            objectNum = int((len(lines)-12) / 7)

            for i in range(objectNum):
                start = 12 + i*7 
                end = 12 + (i+1)*7
                obj_lines = lines[start:end]
                img.addObject(Object(obj_lines))
            imgs.append(img)
    annotation_list.close()
    pos_list.close()
    return imgs, neg_image_paths


if __name__ == '__main__':    

    imgs, neg_image_filenames = parseDataset()
    numObjects = 0
    for img in imgs:
        numObjects += len(img.objects)
    print('Total images: ', len(imgs), ' with a total of ', numObjects, ' objects')

