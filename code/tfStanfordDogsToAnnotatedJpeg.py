import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image, ImageDraw
import PIL
import sys
import os

outPath = sys.argv[1]
print("Will export the training DS to {0}".format(outPath))

if not os.path.exists(outPath):
    os.mkdir(outPath)
idx = 0
for ex in tfds.load('stanford_dogs', split='train'):
    bboxes = ex["objects"]["bbox"].numpy()
    bboxesCount,_ = bboxes.shape
    fname = os.path.basename(ex["image/filename"].numpy().decode('utf-8'))
    npImage = ex["image"].numpy()
    h,w,channels = npImage.shape
    print("shape is {0}".format(npImage.shape))
    image = Image.fromarray(npImage)
    img1 = ImageDraw.Draw(image)
    for i in range(bboxesCount):
        bboxn = np.squeeze(bboxes[i,:]) #  hmin,wmin,hmax,wmax
        (xmin,ymin,xmax,ymax) = w*bboxn[1], h*bboxn[0], w*bboxn[3], h*bboxn[2]
        # print("{0} {1}; {2}; {3}".format(h,w,bboxn, (xmin,ymin,xmax,ymax)))
        img1.rectangle([(xmin,ymin),(xmax,ymax)], outline ="red", width=2) 
    outFilePath = os.path.join(outPath,fname)
    image.save(outFilePath) 
    print("processed {0}\t({1}):\tbbox count {2}".format(fname, idx, bboxesCount))
    idx += 1
print("Done")