import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image, ImageDraw
import PIL
import sys
import os

outAnnotationPath = sys.argv[1]
outDirPath = sys.argv[2]
split = sys.argv[3]
print("Will export the annotations to {0}".format(outAnnotationPath))
print("Will export images to {0}".format(outDirPath))

idx = 0
with open(outAnnotationPath, 'w') as outFile:
    for ex in tfds.load('stanford_dogs', split=split):
        bboxes = ex["objects"]["bbox"].numpy()
        bboxesCount,_ = bboxes.shape
        fname = os.path.basename(ex["image/filename"].numpy().decode('utf-8'))
        npImage = ex["image"].numpy()
        h,w,channels = npImage.shape
        image = Image.fromarray(npImage)
        outFilePath = os.path.join(outDirPath,fname)
        image.save(outFilePath)
        bboxStrs = []
        for i in range(bboxesCount):
            bboxn = np.squeeze(bboxes[i,:]) #  hmin,wmin,hmax,wmax
            (xmin,ymin,xmax,ymax) = int(round(w*bboxn[1])), int(round(h*bboxn[0])), int(round(w*bboxn[3])), int(round(h*bboxn[2]))
            # print("{0} {1}; {2}; {3}".format(h,w,bboxn, (xmin,ymin,xmax,ymax)))
            bboxStrs.append("{0},{1},{2},{3},0".format(xmin,ymin,xmax-xmin+1, ymax-ymin+1))
        bboxStr = " ".join(bboxStrs)
        outFile.write('{0} {1}\n'.format(fname,bboxStr))
        idx += 1
        print("{0}\t:{1} is ready".format(idx,fname))

print("Done")