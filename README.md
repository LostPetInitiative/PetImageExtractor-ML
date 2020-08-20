# Experiments workspace to extract pets from photos

Right now we use [YoloV4 tensorflow implementation](https://github.com/hunglc007/tensorflow-yolov4-tflite) trained on COCO dataset to detect dogs and cats.
Corresponding bounding boxes are extracted.

## Dog detection performance

The average precision (AP) of the dog detection is 91.96%

![Dog detection Average Precision](https://github.com/LostPetInitiative/tensorflow-yolov4-tflite/blob/petExtractor/mAP/results_yolo4_orig_dogs/classes/dog.png)

We assess the average precision on the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (with mixed-in cats from [CatsVsDogs dataset](https://www.kaggle.com/karakaggle/kaggle-cat-vs-dog-dataset) to harden detection)

## Cat detection performance

todo
