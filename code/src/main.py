import os
import sys
import random
import math
import numpy as np
import skimage.io
from skimage import img_as_uint

# Root directory of the project as string
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
                           # sys.path is a list of paths for
                           # interpreter to look for modules 

from mrcnn import utils # common uitility functions and class
import mrcnn.model as modellib # Main Mask R-CNN model implementation.
#from mrcnn import visualize # Display and Visualization Functions.

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco # configurations and data loading code for MS COCO

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Directory to save script output
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/")
print(OUTPUT_DIR)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# Create model object in inference mode (not training)
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# RUN OBJECT DETECTION
image = skimage.io.imread(os.path.join(IMAGE_DIR, "bedroom.jpg"))

# Run detection
results = model.detect([image], verbose=1)
"""Runs the detection pipeline.

images: List of images, potentially of different sizes.

Returns a list of dicts, one dict per image. The dict contains:
rois: [N, (y1, x1, y2, x2)] detection bounding boxes
class_ids: [N] int class IDs
scores: [N] float probability scores for the class IDs
masks: [H, W, N] instance binary masks
"""

# Visualize results
r = results[0]
binary_masks = r["masks"]
binary_masks = np.invert(binary_masks)
n=0
for i in r["class_ids"]:
    print(class_names[i])
    skimage.io.imsave(OUTPUT_DIR+str(n)+".png", img_as_uint(binary_masks[:,:,n]))
    n +=1


