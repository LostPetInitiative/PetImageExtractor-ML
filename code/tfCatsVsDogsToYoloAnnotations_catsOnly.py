import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image, ImageDraw
import PIL
import sys
import os

outAnnotationPath = sys.argv[1]
outDirPath = sys.argv[2]
print("Will export the annotations to {0}".format(outAnnotationPath))
print("Will export images to {0}".format(outDirPath))

idx = 0
with open(outAnnotationPath, 'w') as outFile:
    for ex in tfds.load('cats_vs_dogs', split='train'):
        fname = os.path.basename(ex["image/filename"].numpy().decode('utf-8'))
        npImage = ex["image"].numpy()
        label = ex["label"].numpy()
        if label == 1: # we skip dogs
            continue
        h,w,channels = npImage.shape
        image = Image.fromarray(npImage)
        outFilePath = os.path.join(outDirPath,fname)
        image.save(outFilePath)

        outFile.write('{0} \n'.format(fname))
        idx += 1
        print("{0}:\t{1} is ready".format(idx,fname))

print("Done")